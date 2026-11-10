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
    parser.add_argument("--experiment", help="experiment name being run. Sets the test train splits." \
                        "Default is leave one out.", default="leave_one_out")
    parser.add_argument("--n_cpus", help="n cpus", default=1, type=int)

    args = parser.parse_args()

    n_latent = 50
    hidden_layers =  [512, 128, 64]
    batch_size = 512
    time_weight = 1.0
    kl_weight = 0.00025 * n_latent
    learning_rate = 0.005
    n_epochs = 100
    time_noise = 1

    anndata = sc.read_h5ad(args.infile)
    anndata.X = scipy.sparse.csr_matrix.log1p(anndata.X)

    if args.experiment in ['leave_one_out', 'out_dist']:
        run_on_splits(anndata, args.time_col, args.time_col, n_latent, hidden_layers,
                      batch_size, time_weight, kl_weight, learning_rate, n_epochs, n_cpus=args.n_cpus, time_noise=time_noise)
    elif args.experiment in ['in_dist', 'cross_val']:
        run_on_splits(anndata, args.time_col, args.split_col, n_latent, hidden_layers, batch_size,
                      time_weight, kl_weight, learning_rate, n_epochs,  n_cpus=args.n_cpus, time_noise=time_noise)
    elif args.experiment in ['all', 'train', 'all_train']:
        run_on_all(anndata, args.time_col, n_latent, hidden_layers, batch_size, time_weight,
                   kl_weight, learning_rate, n_epochs, n_cpus=args.n_cpus, time_noise=time_noise)
    elif args.experiment in ['test_train']:
        run_on_test_train(anndata, args.time_col, args.split_col, n_latent, hidden_layers,
                          batch_size, time_weight, kl_weight, learning_rate, n_epochs, n_cpus=args.n_cpus, time_noise=time_noise)
    else:
        raise ValueError(f'Experiment {args.experiment} not recognized.')
        


def run_on_splits(anndata, time_col, split_col, *args, n_cpus=1, time_noise=1):
    results_dir = './'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    unique_splits = anndata.obs[split_col].unique()

    all_fold_dfs = []
    for split in unique_splits:
        print(f"Processing fold: {split}")
        fold_df = process_fold(split, split_col, anndata, time_col, *args, n_cpus=n_cpus,
                               results_dir=results_dir, time_noise=time_noise)
        all_fold_dfs.append(fold_df)
        gc.collect()
        torch.cuda.empty_cache()

    kFold_test_result_df = pd.concat(all_fold_dfs, axis=0)
    kFold_test_result_df.to_csv(f'{results_dir}results.csv', index=True)

    return kFold_test_result_df


def run_on_test_train(anndata, time_col, split_col, *args, n_cpus=1, time_noise=1):
    results_dir = './'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    test_train_df = process_fold('test', split_col, anndata, time_col, *args, n_cpus=n_cpus,
                                  results_dir=results_dir, time_noise=time_noise, return_train=True)

    test_train_df.to_csv(f'{results_dir}results.csv', index=True)
    gc.collect()
    torch.cuda.empty_cache()


def run_on_all(anndata, time_col, n_latent, hidden_layers, batch_size, time_weight, kl_weight,
               learning_rate, n_epochs, n_cpus=1, hv=False, time_noise=1):
    
    workers = min(int(n_cpus), 8)
    workers = max(workers, 0)
    use_persistent = False

    results_dir = './'
    if hv:
        hvgs = sc.pp.highly_variable_genes(anndata, n_top_genes=2000, subset=True, inplace=False).index
        anndata = anndata[:, hvgs].copy()

    y_stars, new_edges, time_levels, old_edges, y_stars_df = string_to_splits(anndata.obs[time_col])
    anndata.obs['time_levels'] = time_levels.cpu()
    anndata.obs['time_levels'] = anndata.obs['time_levels'].astype('int')
    anndata.uns['y_stars'] = y_stars
    anndata.uns['new_edges'] = new_edges

    all_data = pytorch_data.RNA_AnnDataset(anndata, meta_col_names=['time_levels'])


    train, val = all_data.split([0.95, 0.05])

    n_genes = train.anndata.shape[1]
    ga_ord = time_vae.Time_Vae_ord(torch.tensor(y_stars), new_edges, n_genes, n_latent,
        hidden_dims=hidden_layers, kl_weight=kl_weight, time_weight=time_weight, learning_rate=learning_rate,
        time_noise=time_noise)

    
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
        
    ga_ord.to('cuda')

    logger = CSVLogger(results_dir, flush_logs_every_n_steps=log_every_n_steps)

    trainer = L.Trainer(accelerator="gpu", devices=find_usable_cuda_devices(1),
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)], enable_progress_bar=False,
                            default_root_dir=results_dir, logger=logger, log_every_n_steps=log_every_n_steps,
                            max_epochs=n_epochs, precision="32")
    
    trainer.fit(model=ga_ord, train_dataloaders=train_loader, val_dataloaders=val_loader)
    ga_ord.eval()
    

    # DataLoader for the full dataset
    all_loader = torch.utils.data.DataLoader(all_data, batch_size=batch_size, shuffle=False,
                                             collate_fn=pytorch_data.ann_time_collate)

    # Use the same trainer to run prediction (will use GPU as configured)
    # trainer.predict returns whatever ga_ord.predict_step returns (one item per batch).
    # To be explicit and get the regression head output (predicted time), run encode+time_regressor over the DataLoader.
    device = next(ga_ord.parameters()).device
    ga_ord.eval()
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
            mu, _ = ga_ord.encode(dense_x)               # encode -> (mu, sigma)
            pred_time = ga_ord.time_regressor(mu)        # regressor on mu
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
                 log_every_n_steps=50, time_noise=1, return_train=False):

    workers = min(int(n_cpus), 8)
    workers = max(workers, 0)
    use_persistent = False

    subFold_dir = f'{results_dir}/{split}/'
    if not os.path.exists(subFold_dir):
        os.makedirs(subFold_dir)

    y_stars, new_edges, time_levels, old_edges, y_stars_df = string_to_splits(anndata.obs[time_col])
    anndata.obs['time_levels'] = time_levels.cpu()
    anndata.obs['time_levels'] = anndata.obs['time_levels'].astype('int')
    anndata.uns['y_stars'] = y_stars
    anndata.uns['new_edges'] = new_edges

    test_ann = anndata[anndata.obs[split_col] == split, :].copy()
    train_ann_full = anndata[anndata.obs[split_col] != split, :].copy()
    train_ann = train_ann_full.copy()
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
    ga_ord = time_vae.Time_Vae_ord(torch.tensor(y_stars), new_edges, n_genes, n_latent,
                                       hidden_dims=hidden_layers, kl_weight=kl_weight,
                                       time_weight=time_weight, learning_rate=learning_rate, time_noise=time_noise)
    ga_ord.to('cuda')

    logger = CSVLogger(subFold_dir, flush_logs_every_n_steps=log_every_n_steps)

    trainer = L.Trainer(accelerator="gpu", devices=find_usable_cuda_devices(1),
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)],
                enable_progress_bar=False, default_root_dir=subFold_dir, logger=logger, 
                log_every_n_steps=log_every_n_steps, max_epochs=n_epochs, precision="32")
    
    trainer.fit(model=ga_ord, train_dataloaders=train_loader, val_dataloaders=val_loader)

    ga_ord.eval()

    device = next(ga_ord.parameters()).device
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
            mu, _ = ga_ord.encode(dense_x)               # encode -> (mu, sigma)
            pred_time = ga_ord.time_regressor(mu)        # regressor on mu
            all_preds.append(pred_time.detach().cpu().numpy())

    all_pred_times = np.concatenate(all_preds, axis=0).squeeze()
    test_ann.obs['pred_time'] = all_pred_times

    test_result_df = test_ann.obs.loc[:, [split_col, 'pred_time', time_col]]
    test_result_df.columns = ['split', 'pred_time','gt_window']

    if return_train:
        # Predict on the full training set
        if hv:
            train_ann_full = train_ann_full[:, train_ann.var_names].copy()
        train_full_data = pytorch_data.RNA_AnnDataset(train_ann_full, meta_col_names=['time_levels'])
        train_full_loader = torch.utils.data.DataLoader(train_full_data, batch_size=batch_size, shuffle=False,
            collate_fn=pytorch_data.ann_time_collate, num_workers=workers,
            persistent_workers=use_persistent, pin_memory=True)
        train_preds = []
        with torch.no_grad():
            for batch in train_full_loader:
                if isinstance(batch, (list, tuple)):
                    dense_x = batch[0]
                elif isinstance(batch, dict):
                    dense_x = batch.get('dense_x', batch.get('x', None))
                else:
                    dense_x = batch
                if dense_x is None:
                    raise RuntimeError("Could not find dense_x in batch; check your collate_fn output.")
                dense_x = dense_x.to(device)
                mu, _ = ga_ord.encode(dense_x)
                pred_time = ga_ord.time_regressor(mu)
                train_preds.append(pred_time.detach().cpu().numpy())
        train_pred_times = np.concatenate(train_preds, axis=0).squeeze()
        train_ann_full.obs['pred_time'] = train_pred_times
        train_result_df = train_ann_full.obs.loc[:, [split_col, 'pred_time', time_col]]
        train_result_df.columns = ['split', 'pred_time','gt_window']
        train_result_df['split'] = 'train'
        kFold_test_result_df = pd.concat([test_result_df, train_result_df], axis=0)
    else:
        kFold_test_result_df = test_result_df

    return kFold_test_result_df

if __name__ == "__main__":
    main()
