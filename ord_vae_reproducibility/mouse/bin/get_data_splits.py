import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import median_abs_deviation
import argparse
import os
import glob
import anndata as ad

import sys
sys.path.append("/mnt/mr01-data01/j72687wm/")

import vae_models
from vae_models.models import time_vae
from vae_models.data import anndata as pytorch_data
from vae_models.utils import string_to_splits_np, str_to_ord_levels

def get_windowed_edges(adata, numeric_col, new_col):
    unique = pd.unique(adata.obs[numeric_col])
    unique.sort()

    to_cat = {}
    for i, va in enumerate(unique):
        if i == 0:
            max_val = (va + unique[i + 1])/2
            min_val = va - (max_val - va)
        elif i == len(unique) -1:
            min_val = (va + unique[i - 1])/2
            max_val = va + (va - min_val)
        else:
            min_val = (va + unique[i - 1])/2
            max_val = (va + unique[i + 1])/2
        to_cat[float(str(va).rstrip('0').rstrip('.'))] = str(min_val) + '_' + str(max_val)

    adata.obs[new_col] = adata.obs[numeric_col].map(to_cat)
    adata.obs[new_col + '_mean'] = adata.obs[new_col].str.split('_').apply(lambda x: (float(x[0]) + float(x[1]))/2).astype('float')
    return adata

def get_coarser_windows(adata, input_windows, input_num, output_col):
    y_stars, exp1_edges, time_levels = string_to_splits_np(adata.obs[input_windows])
    exp2_edges = []
    for i, edge in enumerate(exp1_edges):
        if i%2 == 1:
            continue
        if i < len(exp1_edges) - 1:
            exp2_edges.append([edge[0], exp1_edges[i + 1, 1]])
        else:
            exp2_edges.append([edge[0], edge[1]])

    exp2_edges = np.array(exp2_edges) 
    raveled_edges = np.unique(exp2_edges)

    exp2_day_index = np.digitize(adata.obs[input_num], raveled_edges, right=True) - 1
    exp2_window = exp2_edges[exp2_day_index]
    adata.obs[output_col] = np.array(['_'.join(map(str, row)) for row in exp2_window])
    output_col_means = output_col + '_mean'
    adata.obs[output_col_means] = adata.obs[output_col].str.split('_').apply(lambda x: (float(x[0]) + float(x[1]))/2).astype('float')

    return adata


def get_coarser_windows_overlapping(adata, input_windows, input_num, output_col):
    y_stars, exp1_edges, time_levels = string_to_splits_np(adata.obs[input_windows])
    overlapping_edges = []

    for i, edge in enumerate(exp1_edges):
        if i + 1 < len(exp1_edges):
            overlapping_edges.append([edge[0], exp1_edges[i + 1][1]])
        else:
            overlapping_edges.append([edge[0], edge[1]])

    overlapping_edges = np.array(overlapping_edges)
    unique_days = pd.unique(adata.obs[input_num])

    for day in unique_days:
        sub = adata.obs.loc[adata.obs['day_num'] == day,:].copy()
        candidate_edges = list(overlapping_edges[[l < float(day) < r for l,r in overlapping_edges]])
        candidate_edges = np.array(['_'.join(map(str, row)) for row in candidate_edges])
        adata.obs.loc[sub.index, output_col] =  np.random.choice(candidate_edges, size=len(sub))
    
    output_col_mean = output_col + '_mean'
    adata.obs[output_col_mean] = adata.obs[output_col].str.split('_').apply(lambda x: (float(x[0]) + float(x[1]))/2).astype('float')
    return adata

def is_outlier_by_batch(adata, metric: str, nmads: int, batch_col: str | None = None):
    if batch_col is not None:
        outlier = pd.Series(False, index=adata.obs.index, dtype=bool)
        for batch in pd.unique(adata.obs[batch_col]):
            subset = adata[adata.obs[batch_col] == batch]
            M = subset.obs[metric]
            outlier[subset.obs.index] = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
                        np.median(M) + nmads * median_abs_deviation(M) < M)
    else:
        M = adata.obs[metric]
        outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
            np.median(M) + nmads * median_abs_deviation(M) < M
        )
    return outlier


def preprocess_data(adata, batch_col='experimental_batch'):
    adata.var["mt"] = adata.var_names.str.startswith("mt:")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=[20], log1p=True)

    adata.obs["outlier"] = (
    is_outlier_by_batch(adata, "log1p_total_counts", 5, batch_col=batch_col)
    | is_outlier_by_batch(adata, "log1p_n_genes_by_counts", 5, batch_col=batch_col)
    | is_outlier_by_batch(adata, "pct_counts_in_top_20_genes", 5, batch_col=batch_col)
    )

    adata.obs["mt_outlier"] = is_outlier_by_batch(adata, "pct_counts_mt", 3, batch_col=batch_col) | (
      adata.obs["pct_counts_mt"] > 8
    )
    adata_cell_filter = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
    
    print('\n\n day value counts:')
    print(adata_cell_filter.obs['day_num'].value_counts())
    
    #exp1 num 'day_num', windows 'windowed_day_exp1'
    adata_cell_filter = get_windowed_edges(adata_cell_filter, 'day_num', 'windowed_day_base')
    
    #exp2 num 'windowed_day_exp2_mean', windows 'windowed_day_exp2'
    adata_cell_filter = get_coarser_windows(adata_cell_filter, 'windowed_day_base', 'day_num', 'windowed_day_coarsened')

    #exp3 num 'windowed_day_exp3_mean', windows 'windowed_day_exp3'get_coarser_windows_overlapping
    adata_cell_filter = get_coarser_windows_overlapping(adata_cell_filter, 'windowed_day_base', 'day_num', 'windowed_day_overlapping')
    return adata_cell_filter


def test_train_split(adata, random_state):
    '''Set max time window as test and a middle time point as test. Then 5% of the remaining data is randomly set as test.'''

    adata.obs['test_train'] = 'train'
    adata.obs.loc[adata.obs['day_num'] == adata.obs['day_num'].max(), 'test_train'] = 'test'
    adata.obs.loc[adata.obs['day_num'] == 10.0, 'test_train'] = 'test'
    adata.obs.loc[adata.obs['day_num'] == 13.0, 'test_train'] = 'test'
    adata.obs.loc[adata.obs['day_num'] == 15.0, 'test_train'] = 'test'
    adata.obs.loc[adata.obs['day_num'] == 17.0, 'test_train'] = 'test'
    remaining = adata.obs['test_train'] == 'train'
    np.random.seed(random_state)
    adata.obs.loc[remaining, 'test_train'] = np.where(np.random.rand(remaining.sum()) < 0.05, 'test', 'train')
    return adata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into training and testing sets.")
    parser.add_argument("--input_h5ad", type=str, required=True, help="Path to the directory containing input h5ad file.")
    parser.add_argument("--meta", type=str, default=None, help="Path to metadata csv")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--out_file", type=str, default="wadot_processed.h5ad", help="Output file for the processed data.")
    parser.add_argument("--is_test", action='store_true', help="If set, run in test mode with smaller data.")
    
    args = parser.parse_args()

    mouse = sc.read_h5ad(args.input_h5ad)

    if args.meta is not None:
        print(f'loading metadata from {args.meta}')
        meta = pd.read_csv(args.meta, index_col='cell')
        meta.columns.difference(mouse.obs.columns)
        tmp_join = pd.merge(mouse.obs, meta.loc[mouse.obs.index, meta.columns.difference(mouse.obs.columns)],
                            how='left', left_index=True, right_index=True)
        mouse.obs = tmp_join
    
    mouse.obs['day_num'] = mouse.obs['day'].str[1:].astype('float')
    mouse.obs.loc[mouse.obs['day'].str.startswith('P'), 'day_num'] = mouse.obs['day_num'].max() + 1 # Set post natal time point as max + 1
    mouse = preprocess_data(mouse)

    mouse = test_train_split(mouse, args.random_state)

    # Select top 5000 highly variable genes to reduce memory usage
    print(f'Selecting top 5000 highly variable genes from {mouse.n_vars} genes...')
    sc.pp.highly_variable_genes(mouse, n_top_genes=5000, subset=True, flavor='seurat_v3')
    print(f'After HVG selection: {mouse.n_vars} genes')

    if args.is_test:
        print('test run on subset, keeping only 4000 cells and 500 hvgs')
        sc.pp.highly_variable_genes(mouse, n_top_genes=500, subset=True, flavor='seurat_v3')
        rand_idx = np.random.choice(mouse.n_obs, size=4000, replace=False)
        mouse = mouse[rand_idx, :].copy()
        

    # Out dist splits from time labels
    # All data does not require splits
    mouse.write_h5ad(args.out_file)