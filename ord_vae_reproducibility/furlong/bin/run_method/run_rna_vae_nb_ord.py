import torch
import gc
import lightning as L
import sys
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Callback

sys.path.append("/mnt/mr01-data01/j72687wm/")

import vae_models
import pickle as pkl
import numpy as np

from vae_models.models import time_vae
from vae_models.data import anndata as pytorch_data
from vae_models.utils import string_to_splits, str_to_ord_levels
from scipy.sparse import csr_matrix
import argparse
import scipy
import pandas as pd
import os
import argparse
import scanpy as sc
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as _np

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="file loc of h5ad file", required=True)
    parser.add_argument("--time_col", help="column name in anndata.obs", required=True)
    parser.add_argument("--split_col", help="column name in anndata.obs for splitting. Only required for in_dist",
                        required=False)
    parser.add_argument("--experiment", help="experiment name being run. Sets the test train splits. Default is leave one out.", default="leave_one_out")
    parser.add_argument("--n_cpus", help="n cpus", default=1, type=int)
    parser.add_argument("--num_workers", help="number of workers for dataloaders", default=0, type=int)
    parser.add_argument("--checkpoint_dir", help="Directory to save checkpoints for resumability. If not set, uses ./checkpoints/", default=None)

    args = parser.parse_args()

    n_latent = 50
    hidden_layers =  [256, 128, 64]
    batch_size = 1024
    time_weight = 1
    kl_weight = 0.00025 * n_latent
    learning_rate = 0.005
    n_epochs = 100
    time_noise = 1.0

    anndata = sc.read_h5ad(args.infile)
    anndata.X = scipy.sparse.csr_matrix.log1p(anndata.X)
    anndata.obs['batch'] = -1

    if args.experiment in ['leave_one_out', 'out_dist']:
        run_on_splits(anndata, args.time_col, args.time_col, n_latent, hidden_layers,
                      batch_size, time_weight, kl_weight, learning_rate, n_epochs, n_cpus=args.n_cpus, num_workers=args.num_workers, time_noise=time_noise, checkpoint_dir=args.checkpoint_dir)
    elif args.experiment in ['in_dist', 'cross_val']:
        run_on_splits(anndata, args.time_col, args.split_col, n_latent, hidden_layers,
                      batch_size, time_weight, kl_weight, learning_rate, n_epochs, n_cpus=args.n_cpus, num_workers=args.num_workers, time_noise=time_noise, checkpoint_dir=args.checkpoint_dir)
    elif args.experiment in ['all', 'train', 'all_train']:
        run_on_all(anndata, args.time_col, n_latent, hidden_layers, batch_size, time_weight, kl_weight,
                   learning_rate, n_epochs, n_cpus=args.n_cpus, num_workers=args.num_workers, time_noise=time_noise)
    else:
        raise ValueError(f'Experiment {args.experiment} not recognized.')
        


def run_on_splits(anndata, time_col, split_col, *args, n_cpus=1, num_workers=0, time_noise=1.0, checkpoint_dir=None):
    results_dir = './'
    # Use provided checkpoint_dir (absolute path for persistence) or fallback to local
    if checkpoint_dir is None:
        checkpoint_dir = f'{results_dir}checkpoints/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    unique_splits = anndata.obs[split_col].unique()

    all_fold_dfs = []
    for split in unique_splits:
        # Sanitize split name for filename
        split_safe = str(split).replace('/', '_').replace(' ', '_')
        checkpoint_file = f'{checkpoint_dir}fold_{split_safe}.csv'
        
        # Check if this fold was already completed (for resume)
        if os.path.exists(checkpoint_file):
            print(f"Resuming: Found checkpoint for fold {split}, loading from {checkpoint_file}")
            fold_df = pd.read_csv(checkpoint_file, index_col=0)
            all_fold_dfs.append(fold_df)
            continue
            
        print(f"Processing fold: {split}")
        fold_df = process_fold(split, split_col, anndata, time_col, *args, n_cpus=n_cpus,
                               results_dir=results_dir, time_noise=time_noise)
        
        # Save checkpoint immediately after completing fold
        fold_df.to_csv(checkpoint_file, index=True)
        print(f"Saved checkpoint: {checkpoint_file}")
        
        all_fold_dfs.append(fold_df)
        gc.collect()
        torch.cuda.empty_cache()

    kFold_test_result_df = pd.concat(all_fold_dfs, axis=0)
    kFold_test_result_df.to_csv(f'{results_dir}results.csv', index=True)

    return kFold_test_result_df


def run_on_all(anndata, time_col, n_latent, hidden_layers, batch_size, time_weight, kl_weight,
               learning_rate, n_epochs, n_cpus=1, num_workers=0, hv=False, time_noise=1.0):
    
    results_dir = './'
    if hv:
        hvgs = sc.pp.highly_variable_genes(anndata, n_top_genes=2000, subset=True, inplace=False).index
        anndata = anndata[:, hvgs].copy()

    y_stars, new_edges, time_levels, old_edges, y_stars_df = string_to_splits(anndata.obs[time_col])
    anndata.obs['time_levels'] = time_levels.cpu()
    anndata.obs['time_levels'] = anndata.obs['time_levels'].astype('int')
    anndata.uns['y_stars'] = y_stars
    anndata.uns['new_edges'] = new_edges
    all_data = pytorch_data.RNA_AnnDataset(anndata, meta_col_names=['time_levels', 'batch'])


    train, val = all_data.split([0.95, 0.05])

    n_genes = train.anndata.shape[1]
    nb_ord = time_vae.Time_Vae_ord_nb(torch.tensor(y_stars), new_edges, n_genes, n_latent,
                                       hidden_dims=hidden_layers, kl_weight=kl_weight,
                                       time_weight=time_weight, learning_rate=learning_rate, time_noise=time_noise)
    
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
        collate_fn=pytorch_data.ann_time_collate, num_workers=num_workers, persistent_workers=(num_workers > 0))

    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False,
        collate_fn=pytorch_data.ann_time_collate, num_workers=num_workers, persistent_workers=(num_workers > 0))
    
    if (train.anndata.shape[0]/batch_size) < 50:
            log_every_n_steps=np.floor(train.anndata.shape[0]/batch_size) - 1
    else:
        log_every_n_steps = 50

    nb_ord.to('cuda')

    logger = CSVLogger(results_dir, flush_logs_every_n_steps=log_every_n_steps)

    trainer = L.Trainer(accelerator="gpu", devices=find_usable_cuda_devices(1),
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)], enable_progress_bar=False,
                            default_root_dir=results_dir, logger=logger, log_every_n_steps=log_every_n_steps,
                            max_epochs=n_epochs)
    
    trainer.fit(model=nb_ord, train_dataloaders=train_loader, val_dataloaders=val_loader)
    nb_ord.eval()
    

    # DataLoader for the full dataset
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=batch_size, shuffle=False,
                                             collate_fn=pytorch_data.ann_time_collate, num_workers=num_workers, persistent_workers=(num_workers > 0))

    # Use the same trainer to run prediction (will use GPU as configured)
    # trainer.predict returns whatever nb_ord.predict_step returns (one item per batch).
    # To be explicit and get the regression head output (predicted time), run encode+time_regressor over the DataLoader.
    device = next(nb_ord.parameters()).device
    nb_ord.eval()
    all_preds = []
    with torch.no_grad():
        for batch in all_loader:
            # adapt to your collate output: common pattern is (dense_x, ...)
            if isinstance(batch, (list, tuple)):
                dense_x = batch[0]
                batch_val = batch[1]['batch']
            elif isinstance(batch, dict):
                dense_x = batch.get('dense_x', batch.get('x', None))
            else:
                dense_x = batch
            if dense_x is None:
                raise RuntimeError("Could not find dense_x in batch; check your collate_fn output.")
            dense_x = dense_x.to(device)
            mu, _ = nb_ord.encode(dense_x, batch_val)               # encode -> (mu, sigma)
            pred_time = nb_ord.time_regressor(mu)        # regressor on mu
            all_preds.append(pred_time.detach().cpu().numpy())

    all_pred_times = np.concatenate(all_preds, axis=0).squeeze()
    all_anndata = all_data.anndata.copy()
    all_anndata.obs['pred_time'] = all_pred_times

    all_anndata.obs['split'] = 'all_data'
    kFold_test_result_df = all_anndata.obs.loc[:, ['split', 'pred_time', time_col]]
    kFold_test_result_df.columns = ['split', 'pred_time','gt_window']
    kFold_test_result_df.to_csv(f'{results_dir}results.csv', index=True)

def process_fold(split, split_col, anndata, time_col, n_latent, hidden_layers, batch_size,
                 time_weight, kl_weight, learning_rate, n_epochs, n_cpus=1, num_workers=0, hv=False, results_dir='./',
                 log_every_n_steps=50, time_noise=1.0):

    subFold_dir = f'{results_dir}/{split}/'
    if not os.path.exists(subFold_dir):
        os.makedirs(subFold_dir)

    y_stars, new_edges, time_levels, old_edges, y_stars_df = string_to_splits(anndata.obs[time_col])
    anndata.obs['time_levels'] = time_levels.cpu()
    anndata.obs['time_levels'] = anndata.obs['time_levels'].astype('int')
    anndata.uns['y_stars'] = y_stars
    anndata.uns['new_edges'] = new_edges

    test_ann = anndata[anndata.obs[split_col] == split, :].copy()
    train_ann = anndata[anndata.obs[split_col] != split, :].copy()
    if hv:
        hvgs = sc.pp.highly_variable_genes(train_ann, n_top_genes=2000, subset=True, inplace=False).index
        train_ann = train_ann[:, hvgs].copy()
        test_ann = test_ann[:, hvgs].copy()

    train = pytorch_data.RNA_AnnDataset(train_ann, meta_col_names=['time_levels', 'batch'])
    test = pytorch_data.RNA_AnnDataset(test_ann, meta_col_names=['time_levels', 'batch'])
    train, val = train.split([0.95, 0.05])

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
        collate_fn=pytorch_data.ann_time_collate, num_workers=num_workers, persistent_workers=(num_workers > 0))

    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False,
        collate_fn=pytorch_data.ann_time_collate, num_workers=num_workers, persistent_workers=(num_workers > 0))

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
        collate_fn=pytorch_data.ann_time_collate, num_workers=num_workers, persistent_workers=(num_workers > 0))
    
    n_genes = train.anndata.shape[1]
    nb_ord = time_vae.Time_Vae_ord_nb(torch.tensor(y_stars), new_edges, n_genes, n_latent,
                        hidden_dims=hidden_layers, kl_weight=kl_weight,
                        time_weight=time_weight, learning_rate=learning_rate, time_noise=time_noise)
    nb_ord.to('cuda')

    logger = CSVLogger(subFold_dir, flush_logs_every_n_steps=log_every_n_steps)

    trainer = L.Trainer(accelerator="gpu", devices=find_usable_cuda_devices(1),
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)],
                enable_progress_bar=False, default_root_dir=subFold_dir, logger=logger, 
                log_every_n_steps=log_every_n_steps, max_epochs=n_epochs)

    trainer.fit(model=nb_ord, train_dataloaders=train_loader, val_dataloaders=val_loader)

    nb_ord.eval()

    device = next(nb_ord.parameters()).device
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            # adapt to your collate output: common pattern is (dense_x, ...)
            if isinstance(batch, (list, tuple)):
                dense_x = batch[0]
                batch_val = batch[1]['batch']
            elif isinstance(batch, dict):
                dense_x = batch.get('dense_x', batch.get('x', None))
            else:
                dense_x = batch
            if dense_x is None:
                raise RuntimeError("Could not find dense_x in batch; check your collate_fn output.")
            dense_x = dense_x.to(device)
            mu, _ = nb_ord.encode(dense_x, batch_val)              # encode -> (mu, sigma)
            pred_time = nb_ord.time_regressor(mu)        # regressor on mu
            all_preds.append(pred_time.detach().cpu().numpy())

    all_pred_times = np.concatenate(all_preds, axis=0).squeeze()
    test_ann.obs['pred_time'] = all_pred_times

    kFold_test_result_df = test_ann.obs.loc[:, [split_col, 'pred_time', time_col]]
    kFold_test_result_df.columns = ['split', 'pred_time','gt_window']
    return kFold_test_result_df

if __name__ == "__main__":
    main()