import os
import sys

# necessary: change current path to TemporalVAE
# if os.getcwd().split("/")[-1] != "TemporalVAE":
#     os.chdir("..")
sys.path.append('/mnt/mr01-data01/j72687wm/ordVae_paper/temporal_vae_benchmark/TemporalVAE/Fig2_TemproalVAE_against_benchmark_methods')
from utils.GPU_manager_pytorch import auto_select_gpu_and_cpu, check_memory
import logging
from utils.logging_system import LogHelper
from utils.utils_DandanProject import trans_time, plot_training_loss_for_tags, denormalize
import anndata
import pandas as pd

import numpy as np
from model_master.experiment import VAEXperiment
from model_master.dataset import SupervisedVAEDataset, SupervisedVAEDataset_onlyPredict
from model_master import vae_models

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pathlib import Path
import argparse
import scanpy as sc
import yaml
import scipy
import random

def run_on_splits(anndata, time_col, split_col, config, *args, n_cpus=1, hv=True, checkpoint_dir=None):
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
        fold_df = process_fold(split, split_col, anndata, time_col, config, *args, n_cpus=n_cpus,
                               results_dir=str(results_dir))
        
        # Save checkpoint immediately after completing fold
        fold_df.to_csv(checkpoint_file, index=True)
        print(f"Saved checkpoint: {checkpoint_file}")
        
        all_fold_dfs.append(fold_df)

    kFold_test_result_df = pd.concat(all_fold_dfs, axis=0)
    kFold_test_result_df.to_csv(results_dir / 'results.csv', index=True)

def process_fold(split, split_col, anndata, time_col, config, n_cpus=1, hv=True, results_dir='./',
                 max_cells=100000, train_epoch_num=100, parameters=None, use_gpu=False, save_model=None):
    
    test = anndata[anndata.obs[split_col] == split].copy()
    train_all = anndata[anndata.obs[split_col] != split].copy()

    if hv:
        hvgs = sc.pp.highly_variable_genes(train_all, n_top_genes=2000, subset=True, inplace=False).index
        train_all = train_all[:, hvgs].copy()
        test = test[:, hvgs].copy()

    subFold_dir = f'{results_dir}/{split}/'
    if not os.path.exists(subFold_dir):
        os.makedirs(subFold_dir)

    n_cells = train_all.shape[0]

    if n_cells > max_cells:
        idx = random.sample(range(n_cells), max_cells)
        train = train_all[idx,:].copy()
    else:
        train = train_all.copy()
        

    #generate train val split
    idx = np.arange(train.shape[0])
    np.random.shuffle(idx)  #inplace
    n_cells = len(idx)
    cum_props = np.cumsum([0.95, 0.05])
    cum_props = np.insert(cum_props, 0, 0)
    edges = (n_cells*cum_props).astype(int)

    val_adata = train[edges[1]:].copy()
    train_adata = train[:edges[1]].copy()

    x_sc_train = torch.tensor(train_adata.X.todense(), dtype=torch.float)
    x_sc_val = torch.tensor(val_adata.X.todense(), dtype=torch.float)
    x_sc_test = torch.tensor(test.X.todense(), dtype=torch.float)

    y_time_train = x_sc_train.new_tensor(np.array(train_adata.obs[time_col]).astype(float))
    y_time_val = x_sc_val.new_tensor(np.array(val_adata.obs[time_col]).astype(float))
    y_time_test = x_sc_test.new_tensor(np.array(test.obs[time_col]).astype(float))

    #  include dummy for batch (In dataloader but not used in the models forward method)
    train_data = [[x_sc_train[i, :], y_time_train[i], torch.tensor(-1)] for i in range(x_sc_train.shape[0])]
    val_data = [[x_sc_val[i, :], y_time_val[i], torch.tensor(-1)] for i in range(x_sc_val.shape[0])]
    test_data = [[x_sc_test[i, :], y_time_test[i], torch.tensor(-1)] for i in range(x_sc_test.shape[0])]

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    # setup logging
    tb_logger = TensorBoardLogger(save_dir=subFold_dir,
                                  name=config['model_params']['name'], )
    # if int(config['model_params']['in_channels']) == 0:
    config['model_params']['in_channels'] = x_sc_train.shape[1]

    data = SupervisedVAEDataset(train_data=train_data, val_data=val_data, predict_data=val_data,
                                train_batch_size=len(train_data), val_batch_size=len(val_data),
                                predict_batch_size=len(val_data))
    
    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])

    experiment = VAEXperiment(MyVAEModel, config['exp_params'])

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # 2023-09-07 20:34:25 add check memory
    check_memory()
    device = auto_select_gpu_and_cpu()
    print("Auto select run on {}".format(device))

    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=train_epoch_num
                     )
    

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    runner.fit(experiment, data)

    # predict on test data, may need to do this in chunks
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    results = runner.predict(experiment, data_test)
    pred_time = results[0][0]

    if (np.isnan(np.squeeze(pred_time)).any()):
            print("The Array contain NaN values")
    else:
        print("The Array does not contain NaN values")

    _result_df = pd.DataFrame(data={'split': test.obs[split_col],
                                'pred_time': np.squeeze(pred_time, axis=1),
                                'gt_window': test.obs[time_col]},
                                index=test.obs.index)

    del MyVAEModel
    del runner
    del experiment
    # 清除CUDA缓存
    torch.cuda.empty_cache()
    return _result_df

    

def run_on_all(anndata, time_col, config, n_cpus=1, hv=True, results_dir='./',
                 max_cells=100000, parameters=None, use_gpu=False, train_epoch_num=100, save_model=None):
    results_dir = './'
    if hv:
        hvgs = sc.pp.highly_variable_genes(anndata, n_top_genes=2000, subset=True, inplace=False).index
        anndata = anndata[:, hvgs].copy()

    n_cells = anndata.shape[0]

    if n_cells > max_cells:
        print(f"Training data has {n_cells} cells, which exceeds max_cells={max_cells}. Subsampling for training.")
        idx = random.sample(range(n_cells), max_cells)
        idx_missing = list(set(range(n_cells)) - set(idx))
        train = anndata[idx,:].copy()
        train_to_predict = anndata[idx_missing,:].copy()
    else:
        train = anndata.copy()
        train_to_predict = None

    # generate train val split
    idx = np.arange(train.shape[0])
    np.random.shuffle(idx)  #inplace
    n_cells = len(idx)
    cum_props = np.cumsum([0.9, 0.1])
    cum_props = np.insert(cum_props, 0, 0)
    edges = (n_cells*cum_props).astype(int)

    val_adata = train[edges[1]:].copy()
    train_adata = train[:edges[1].copy()]
    
    x_sc_train = torch.tensor(train_adata.X.todense(), dtype=torch.float)
    x_sc_val = torch.tensor(val_adata.X.todense(), dtype=torch.float)

    y_time_train = x_sc_train.new_tensor(np.array(train_adata.obs[time_col]).astype(float))
    y_time_val = x_sc_val.new_tensor(np.array(val_adata.obs[time_col]).astype(float))

    #  include dummy for batch (In dataloader but not used in the models forward method)
    train_data = [[x_sc_train[i, :], y_time_train[i], torch.tensor(-1)] for i in range(x_sc_train.shape[0])]
    val_data = [[x_sc_val[i, :], y_time_val[i], torch.tensor(-1)] for i in range(x_sc_val.shape[0])]

     # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    # setup logging
    tb_logger = TensorBoardLogger(save_dir=results_dir,
                                  name=config['model_params']['name'], )
    # if int(config['model_params']['in_channels']) == 0:
    config['model_params']['in_channels'] = x_sc_train.shape[1]

    data = SupervisedVAEDataset(train_data=train_data, val_data=val_data, predict_data=val_data,
                                train_batch_size=len(train_data), val_batch_size=len(val_data),
                                predict_batch_size=len(val_data))
    
    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])

    experiment = VAEXperiment(MyVAEModel, config['exp_params'])

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # 2023-09-07 20:34:25 add check memory
    check_memory()
    device = auto_select_gpu_and_cpu()
    print("Auto select run on {}".format(device))

    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=train_epoch_num
                     )
    

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    runner.fit(experiment, data)

    # Prepare data for prediction (using the full 'train' object which includes train+val splits)
    # We use todense() here assuming the dataset fits in memory (consistent with process_fold)
    x_sc_all = torch.tensor(anndata.X.todense(), dtype=torch.float)
    y_time_all = x_sc_all.new_tensor(np.array(anndata.obs[time_col]).astype(float))
    
    # Create list of [x, y, dummy]
    all_predict_data = [[x_sc_all[i, :], y_time_all[i], torch.tensor(-1)] for i in range(x_sc_all.shape[0])]

    # Create dataset for prediction
    # Using a fixed batch size (e.g., 2048) is safer than full-batch for large datasets
    data_all = SupervisedVAEDataset_onlyPredict(predict_data=all_predict_data, predict_batch_size=2048)
    
    # Run prediction
    results = runner.predict(experiment, data_all)
    
    # Aggregate results from batches
    # results is a list of batch outputs. Based on process_fold, results[i][0] is the pred_time tensor.
    all_preds = []
    for batch_res in results:
        all_preds.append(batch_res[0])
        
    # Concatenate and convert to numpy
    pred_time = np.concatenate(all_preds, axis=0) 

    if (np.isnan(np.squeeze(pred_time)).any()):
            print("The Array contain NaN values")
    else:
        print("The Array does not contain NaN values")

    # Create Results DataFrame
    kFold_test_result_df = pd.DataFrame(data={
        'split': 'all_data',
        'pred_time': np.squeeze(pred_time),
        'gt_window': anndata.obs[time_col].values
    }, index=anndata.obs.index)
    
    # Save to CSV
    print(f"Saving results to {results_dir}results.csv")
    kFold_test_result_df.to_csv(f'{results_dir}results.csv', index=True)
    
    # Cleanup
    del MyVAEModel
    del runner
    del experiment
    torch.cuda.empty_cache()
    # --- PREDICTION LOGIC END ---


    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="file loc of h5ad file")
    parser.add_argument("--time_col", help="column name for time info", default='exp2_centers')
    parser.add_argument("--split_col", help="column name in anndata.obs for splitting. Only required for in_dist",
                        required=False)
    parser.add_argument("--experiment", help="experiment name being run. Sets the test train splits." \
                            "Default is leave one out.", default="leave_one_out")
    parser.add_argument("--max_cells", default=100000, help="max cells to use for training", type=int)
    parser.add_argument("--n_cpus", help="n cpus", default=1, type=int)
    parser.add_argument("--checkpoint_dir", help="Directory to save checkpoints for resumability. If not set, uses ./checkpoints/", default=None)
    args = parser.parse_args()

    with open("/mnt/mr01-data01/j72687wm/ordVae_paper/nextflow/templates/supervise_vae_regressionclfdecoder_exp2_toyDataset.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    anndata = sc.read_h5ad(args.infile)
    anndata.X = scipy.sparse.csr_matrix.log1p(anndata.X)

    if args.experiment in ['leave_one_out', 'out_dist']:
        run_on_splits(anndata, args.time_col, args.time_col, config, n_cpus=args.n_cpus, checkpoint_dir=args.checkpoint_dir)
    elif args.experiment in ['in_dist', 'cross_val']:
        run_on_splits(anndata, args.time_col, args.split_col, config, n_cpus=args.n_cpus, checkpoint_dir=args.checkpoint_dir)
    elif args.experiment in ['all', 'train', 'all_train']:
        run_on_all(anndata, args.time_col, config, n_cpus=args.n_cpus)
    else:
        raise ValueError(f"Unsupported experiment type '{args.experiment}'.")

if __name__ == "__main__":
     main()
