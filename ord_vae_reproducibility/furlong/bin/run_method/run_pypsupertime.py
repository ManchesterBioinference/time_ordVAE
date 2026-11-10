from pypsupertime import Psupertime
import scanpy as sc
import anndata as ad
import random
import argparse
import scipy
import os
import pickle as pkl
import pandas as pd
import time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="file loc of h5ad file", required=True)
    parser.add_argument("--time_col", help="column name in anndata.obs", required=True)
    parser.add_argument("--split_col", help="column name in anndata.obs for splitting. Only required for in_dist",
                        required=False)
    parser.add_argument("--experiment", help="experiment name being run. Sets the test train splits. Default is leave one out.", default="leave_one_out")
    parser.add_argument("--n_cpus", help="n cpus", default=1, type=int)
    parser.add_argument("--num_workers", help="number of workers for dataloaders", default=0, type=int)
    parser.add_argument("--max_cells", help="maximum number of cells to use for training (subsample if exceeded)", default=10000, type=int)
    parser.add_argument("--checkpoint_dir", help="Directory to save checkpoints for resumability. If not set, uses ./checkpoints/", default=None)

    args = parser.parse_args()


    anndata = sc.read_h5ad(args.infile)
    anndata.X = scipy.sparse.csr_matrix.log1p(anndata.X)

    if args.experiment in ['leave_one_out', 'out_dist']:
        run_on_splits(anndata, args.time_col, args.time_col, n_cpus=args.num_workers, max_cells=args.max_cells, checkpoint_dir=args.checkpoint_dir)
    elif args.experiment in ['in_dist', 'cross_val']:
        run_on_splits(anndata, args.time_col, args.split_col, n_cpus=args.num_workers, max_cells=args.max_cells, checkpoint_dir=args.checkpoint_dir)
    elif args.experiment in ['all', 'train', 'all_train']:
        run_on_all(anndata, args.time_col, max_cells=args.max_cells, n_cpus=args.num_workers)
    else:
        raise ValueError(f'Experiment {args.experiment} not recognized.')


def run_on_all(anndata, time_col, max_cells=10000, n_cpus=1, hv=False):
    results_dir = './'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    n_cells = anndata.n_obs

    if hv:
        hvgs = sc.pp.highly_variable_genes(anndata, n_top_genes=2000, subset=True, inplace=False).index
        anndata = anndata[:, hvgs].copy()

    data_train_to_predict = None
    if n_cells > max_cells:
        print(f"Training data has {n_cells} cells, which exceeds max_cells={max_cells}. Subsampling for training.")
        idx = random.sample(range(n_cells), max_cells)
        idx_missing = list(set(range(n_cells)) - set(idx))
        data_train_to_predict = anndata[idx_missing,:].copy()
        train = anndata[idx,:].copy()
    else:
        train = anndata.copy()

    p = Psupertime()
    train = p.run(train, time_col)

    if data_train_to_predict is not None:
        try:
            data_train_to_predict = data_train_to_predict[:, train.var_names].copy()
        except Exception:
            missing = set(train.var_names) - set(data_train_to_predict.var_names)
            raise ValueError(
                f"Test is missing {len(missing)} genes from training features; cannot align."
            )
        p.model.predict_psuper(data_train_to_predict, inplace=True)
        
        all_anndata = ad.concat([train, data_train_to_predict], axis=0)
    else:
        all_anndata = train

    all_anndata.obs['split'] = 'all_data'

    kFold_test_result_df = all_anndata.obs.loc[:, ['split', 'psupertime', time_col]]
    kFold_test_result_df.columns = ['split', 'pred_time','gt_window']
    kFold_test_result_df.to_csv(f'{results_dir}results.csv', index=True)


    time.sleep(5) # Try and ensure all file handles are closed before exiting.


def run_on_splits(anndata, time_col, split_col, n_cpus=1, max_cells=100000, checkpoint_dir=None):
    results_dir = './'
    # Use provided checkpoint_dir (absolute path for persistence) or fallback to local
    if checkpoint_dir is None:
        checkpoint_dir = f'{results_dir}checkpoints/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    all_fold_dfs = []
    for split in pd.unique(anndata.obs[split_col]):
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
        _res_df = process_fold(split, split_col, anndata, time_col, n_cpus, max_cells, hv=False)
        
        # Save checkpoint immediately after completing fold
        _res_df.to_csv(checkpoint_file, index=True)
        print(f"Saved checkpoint: {checkpoint_file}")
        
        all_fold_dfs.append(_res_df)

    kFold_test_result_df = pd.concat(all_fold_dfs, axis=0)
    kFold_test_result_df.to_csv(f'{results_dir}results.csv', index=True)
    time.sleep(5) # Try and ensure all file handles are closed before exiting.


def process_fold(split, split_col, anndata, time_col, n_cpus, max_cells, hv=False):
    test = anndata[anndata.obs[split_col] == split].copy()
    train = anndata[anndata.obs[split_col] != split].copy()
    n_cells = train.n_obs

    if hv:
        hvgs = sc.pp.highly_variable_genes(train, n_top_genes=2000, subset=True, inplace=False).index
        train = train[:, hvgs].copy()
        test = test[:, hvgs].copy()

    if n_cells > max_cells:
        print(f"Training data has {n_cells} cells, which exceeds max_cells={max_cells}. Subsampling for training.")
        idx = random.sample(range(n_cells), max_cells)
        idx_missing = list(set(range(n_cells)) - set(idx))

        train = train[idx,:].copy()
    

    p = Psupertime(n_jobs = n_cpus)
    train = p.run(train, time_col)

    try:
        test = test[:, train.var_names].copy()
    except Exception:
        missing = set(train.var_names) - set(test.var_names)
        raise ValueError(
            f"Test is missing {len(missing)} genes from training features; cannot align."
        )

    test_result_df = p.model.predict_psuper(test, inplace=False)

    return_df = test_result_df.loc[:, [split_col, 'psupertime', time_col]]
    return_df.columns = ['split', 'pred_time','gt_window']

    return return_df

if __name__ == "__main__":
    main()



