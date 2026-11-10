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
from pathlib import Path

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="file loc of h5ad file", required=True)
    parser.add_argument("--time_col", help="column name in anndata.obs", required=True)
    parser.add_argument("--split_col", help="column name in anndata.obs for splitting. Only required for in_dist",
                        required=False)
    parser.add_argument("--experiment", help="experiment name being run. Sets the test train splits. Default is leave one out.", default="leave_one_out")
    parser.add_argument("--n_cpus", help="number of workers to use in dataloader", default=1, type=int)
    parser.add_argument("--checkpoint_dir", help="Directory to save checkpoints for resumability. If not set, uses ./checkpoints/", default=None)

    args = parser.parse_args()

    n_latent = 50
    hidden_layers =  [256, 128, 64]
    batch_size = 1024
    time_weight = 1
    kl_weight = 0.00025 * n_latent
    learning_rate = 0.005
    n_epochs = 100

    anndata = sc.read_h5ad(args.infile)
    anndata.X = scipy.sparse.csr_matrix.log1p(anndata.X)

    if args.experiment in ['leave_one_out', 'out_dist']:
        run_on_splits(anndata, args.time_col, args.time_col, n_latent, hidden_layers,
                      batch_size, time_weight, kl_weight, learning_rate, n_epochs, n_cpus=args.n_cpus, checkpoint_dir=args.checkpoint_dir)
    elif args.experiment in ['in_dist', 'cross_val']:
        run_on_splits(anndata, args.time_col, args.split_col, n_latent, hidden_layers, batch_size,
                      time_weight, kl_weight, learning_rate, n_epochs,  n_cpus=args.n_cpus, checkpoint_dir=args.checkpoint_dir)
    elif args.experiment in ['all', 'train', 'all_train']:
        run_on_all(anndata, args.time_col, n_latent, hidden_layers, batch_size, time_weight,
                   kl_weight, learning_rate, n_epochs, n_cpus=args.n_cpus)
    else:
        raise ValueError(f'Experiment {args.experiment} not recognized.')

def run_on_splits(anndata, time_col, split_col, *args, n_cpus=1, checkpoint_dir=None):
    results_dir = Path('./')
    # Use provided checkpoint_dir (absolute path for persistence) or fallback to local
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else results_dir / 'checkpoints'
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    unique_splits = anndata.obs[split_col].unique()

    all_fold_dfs = []
    for split in unique_splits:
        # Sanitize split name for filename
        split_safe = str(split).replace('/', '_').replace(' ', '_')
        checkpoint_file = checkpoint_dir / f'fold_{split_safe}.csv'
        
        # Check if this fold was already completed (for resume)
        if checkpoint_file.exists():
            print(f"Resuming: Found checkpoint for fold {split}, loading from {checkpoint_file}")
            fold_df = pd.read_csv(checkpoint_file, index_col=0)
            all_fold_dfs.append(fold_df)
            continue
            
        print(f"Processing fold: {split}")
        fold_df = process_fold(split, split_col, anndata, time_col, *args, n_cpus=n_cpus,
                               results_dir=str(results_dir))
        
        # Save checkpoint immediately after completing fold
        fold_df.to_csv(checkpoint_file, index=True)
        print(f"Saved checkpoint: {checkpoint_file}")
        
        all_fold_dfs.append(fold_df)
        gc.collect()
        torch.cuda.empty_cache()

    kFold_test_result_df = pd.concat(all_fold_dfs, axis=0)
    kFold_test_result_df.to_csv(results_dir / 'results.csv', index=True)

    return kFold_test_result_df


def run_on_all(anndata, time_col, n_latent, hidden_layers, batch_size, time_weight, kl_weight,
               learning_rate, n_epochs, n_cpus=1, hv=False):

    workers = min(int(n_cpus), 8)
    workers = max(workers, 0)
    use_persistent = True if workers > 0 else False
    
    results_dir = './'
    if hv:
        hvgs = sc.pp.highly_variable_genes(anndata, n_top_genes=2000, subset=True, inplace=False).index
        anndata = anndata[:, hvgs].copy()

    y_stars, new_edges, time_levels, old_edges, y_stars_df = string_to_splits(anndata.obs[time_col])
    anndata.obs['time_levels'] = time_levels.cpu()
    anndata.obs['time_levels'] = anndata.obs['time_levels'].astype('int')
    anndata.uns['y_stars'] = y_stars
    new_edges = torch.unique(new_edges.flatten())
    anndata.uns['new_edges'] = new_edges

    all_data = pytorch_data.RNA_AnnDataset(anndata, meta_col_names=['time_levels'])


    train, val = all_data.split([0.95, 0.05])

    n_genes = train.anndata.shape[1]
    ga_cum_log = time_vae.Time_Vae_cumulative_logit(torch.tensor(y_stars), new_edges, n_genes, n_latent,
        hidden_dims=hidden_layers, kl_weight=kl_weight, time_weight=time_weight, learning_rate=learning_rate)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
        collate_fn=pytorch_data.ann_time_collate, drop_last=True, num_workers=workers,
        persistent_workers=use_persistent, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False,
        collate_fn=pytorch_data.ann_time_collate, num_workers=workers,
        persistent_workers=use_persistent, pin_memory=True)
    
    if (train.anndata.shape[0]/batch_size) < 50:
            log_every_n_steps=np.floor(train.anndata.shape[0]/batch_size) - 1
    else:
        log_every_n_steps = 50
        
    ga_cum_log.to('cuda')

    logger = CSVLogger(results_dir, flush_logs_every_n_steps=log_every_n_steps)

    trainer = L.Trainer(accelerator="gpu", devices=find_usable_cuda_devices(1),
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)], enable_progress_bar=False,
                            default_root_dir=results_dir, logger=logger, log_every_n_steps=log_every_n_steps,
                            max_epochs=n_epochs)
    
    trainer.fit(model=ga_cum_log, train_dataloaders=train_loader, val_dataloaders=val_loader)
    ga_cum_log.eval()


    # DataLoader for the full dataset
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=batch_size, shuffle=False,
                                             collate_fn=pytorch_data.ann_time_collate)

    # Use the same trainer to run prediction (will use GPU as configured)
    # trainer.predict returns whatever ga_ord.predict_step returns (one item per batch).
    # To be explicit and get the regression head output (predicted time), run encode+time_regressor over the DataLoader.
    device = next(ga_cum_log.parameters()).device
    ga_cum_log.eval()
    all_preds = []
    with torch.no_grad():
        for batch in all_loader:
            # adapt to your collate output: common pattern is (dense_x, ...)
            if isinstance(batch, (list, tuple)):
                dense_x = batch[0]
            elif isinstance(batch, dict):
                dense_x = batch.get('dense_x', batch.get('x', None))
            else:
                dense_x = batch
            if dense_x is None:
                raise RuntimeError("Could not find dense_x in batch; check your collate_fn output.")
            dense_x = dense_x.to(device)
            mu, _ = ga_cum_log.encode(dense_x)               # encode -> (mu, sigma)
            pred_time = ga_cum_log.time_regressor(mu)        # regressor on mu
            all_preds.append(pred_time.detach().cpu().numpy())

    all_pred_times = np.concatenate(all_preds, axis=0).squeeze()
    all_anndata = all_data.anndata.copy()
    all_anndata.obs['pred_time'] = all_pred_times

    all_anndata.obs['split'] = 'all_data'
    kFold_test_result_df = all_anndata.obs.loc[:, ['split', 'pred_time', time_col]]
    kFold_test_result_df.columns = ['split', 'pred_time','gt_window']
    kFold_test_result_df.to_csv(f'{results_dir}results.csv', index=True)


def process_fold(split, split_col, anndata, time_col, n_latent, hidden_layers, batch_size,
                 time_weight, kl_weight, learning_rate, n_epochs, n_cpus=1, hv=False, results_dir='./',
                 log_every_n_steps=50):

    workers = min(int(n_cpus), 8)
    workers = max(workers, 0)
    use_persistent = True if workers > 0 else False

    subFold_dir = f'{results_dir}/{split}/'
    if not os.path.exists(subFold_dir):
        os.makedirs(subFold_dir)

    y_stars, new_edges, time_levels, old_edges, y_stars_df = string_to_splits(anndata.obs[time_col])
    anndata.obs['time_levels'] = time_levels.cpu()
    anndata.obs['time_levels'] = anndata.obs['time_levels'].astype('int')
    anndata.uns['y_stars'] = y_stars
    new_edges = torch.unique(new_edges.flatten())
    anndata.uns['new_edges'] = new_edges

    test_ann = anndata[anndata.obs[split_col] == split, :].copy()
    train_ann = anndata[anndata.obs[split_col] != split, :].copy()
    if hv:
        hvgs = sc.pp.highly_variable_genes(train_ann, n_top_genes=2000, subset=True, inplace=False).index
        train_ann = train_ann[:, hvgs].copy()
        test_ann = test_ann[:, hvgs].copy()

    train = pytorch_data.RNA_AnnDataset(train_ann, meta_col_names=['time_levels'])
    test = pytorch_data.RNA_AnnDataset(test_ann, meta_col_names=['time_levels'])
    train, val = train.split([0.95, 0.05])

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
        collate_fn=pytorch_data.ann_time_collate, drop_last=True, num_workers=workers,
        persistent_workers=use_persistent, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False,
        collate_fn=pytorch_data.ann_time_collate, num_workers=workers,
        persistent_workers=use_persistent, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
        collate_fn=pytorch_data.ann_time_collate, num_workers=workers,
        persistent_workers=use_persistent, pin_memory=True)
    
    n_genes = train.anndata.shape[1]
    ga_cum_log = time_vae.Time_Vae_cumulative_logit(torch.tensor(y_stars), new_edges, n_genes, n_latent,
                                       hidden_dims=hidden_layers, kl_weight=kl_weight,
                                       time_weight=time_weight, learning_rate=learning_rate)
    ga_cum_log.to('cuda')

    logger = CSVLogger(subFold_dir, flush_logs_every_n_steps=log_every_n_steps)

    trainer = L.Trainer(accelerator="gpu", devices=find_usable_cuda_devices(1),
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)],
                enable_progress_bar=False, default_root_dir=subFold_dir, logger=logger, 
                log_every_n_steps=log_every_n_steps, max_epochs=n_epochs)
    
    trainer.fit(model=ga_cum_log, train_dataloaders=train_loader, val_dataloaders=val_loader)

    ga_cum_log.eval()

    device = next(ga_cum_log.parameters()).device
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            # adapt to your collate output: common pattern is (dense_x, ...)
            if isinstance(batch, (list, tuple)):
                dense_x = batch[0]
            elif isinstance(batch, dict):
                dense_x = batch.get('dense_x', batch.get('x', None))
            else:
                dense_x = batch
            if dense_x is None:
                raise RuntimeError("Could not find dense_x in batch; check your collate_fn output.")
            dense_x = dense_x.to(device)
            mu, _ = ga_cum_log.encode(dense_x)               # encode -> (mu, sigma)
            pred_time = ga_cum_log.time_regressor(mu)        # regressor on mu
            all_preds.append(pred_time.detach().cpu().numpy())

    all_pred_times = np.concatenate(all_preds, axis=0).squeeze()
    test_ann.obs['pred_time'] = all_pred_times

    kFold_test_result_df = test_ann.obs.loc[:, [split_col, 'pred_time', time_col]]
    kFold_test_result_df.columns = ['split', 'pred_time','gt_window']
    return kFold_test_result_df


if __name__ == "__main__":
    main()