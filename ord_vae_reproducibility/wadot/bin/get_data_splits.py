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

def output_for_r(adata):
    if not os.path.exists('r_output'):
        os.makedirs('r_output')
    output_file_base = os.path.join('r_output', 'furlong_data')
    adata_no_obs = adata.copy()
    adata_no_obs.obs = adata_no_obs.obs.iloc[:, 0:0]  # keep index only
    adata_no_obs.write(f"{output_file_base}_counts_only.h5ad")

    # 2. Save metadata to CSV
    adata.obs.to_csv(f"{output_file_base}_obs_metadata.csv")
    adata.var.to_csv(f"{output_file_base}_var_metadata.csv")

def load_h5(dir_path):
    adatas = []
    for file in glob.glob(os.path.join(dir_path, '*.h5')):
        filename = os.path.basename(file)
        batch = filename.split('_gene')[0]
        tmp = sc.read_10x_h5(file)
        tmp.var_names_make_unique()
        tmp.obs['batch'] = batch
        tmp.obs.index = tmp.obs.index + '_' + batch
        adatas.append(tmp)

    adata = ad.concat(adatas, axis=0)

    if not adata.obs.index.is_unique:
        raise ValueError("Non-unique indices found in concatenated data")
    return adata

def is_outlier_by_batch(adata, metric: str, nmads: int, batch_col: str | None = None):
    if batch_col is not None:
        outlier = pd.Series(index = adata.obs.index)
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

def in_dist_splits(adata, nfolds, random_state=42):
    np.random.seed(random_state)
    adata.obs['in_dist_split'] = np.random.randint(0, nfolds, size=adata.n_obs)
    return adata

def preprocess_data(adata, batch_col='batch'):
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

    adata_cell_filter_noGDF9 = adata_cell_filter[~adata_cell_filter.obs['batch'].str.contains('GDF9')]
    adata_cell_filter_noGDF9.obs['day'] = adata_cell_filter_noGDF9.obs['batch'].str.split('_').str[1].str[1:]
    adata_cell_filter_noGDF9.obs['treat'] = adata_cell_filter_noGDF9.obs['batch'].str.split('_').str[2]
    adata_cell_filter_noGDF9.obs['dummy'] = adata_cell_filter_noGDF9.obs['batch'].str.split('_').str[3]
    
    print('\n\n day value counts:')
    print(adata_cell_filter_noGDF9.obs['day'].value_counts())

    print('\n\n treat value counts:')
    print(adata_cell_filter_noGDF9.obs['treat'].value_counts())

    print('\n\n dummy value counts:')
    print(adata_cell_filter_noGDF9.obs['dummy'].value_counts())

    adata_cell_filter_noGDF9.obs['day_num'] = adata_cell_filter_noGDF9.obs['day'].replace(to_replace='iPSC', value='19', inplace=False).astype('float')
    
    #exp1 num 'day_num', windows 'windowed_day_exp1'
    adata_cell_filter_noGDF9 = get_windowed_edges(adata_cell_filter_noGDF9, 'day_num', 'windowed_day_exp1')
    
    #exp2 num 'windowed_day_exp2_mean', windows 'windowed_day_exp2'
    adata_cell_filter_noGDF9 = get_coarser_windows(adata_cell_filter_noGDF9, 'windowed_day_exp1', 'day_num', 'windowed_day_exp2')

    #exp3 num 'windowed_day_exp3_mean', windows 'windowed_day_exp3'get_coarser_windows_overlapping
    adata_cell_filter_noGDF9 = get_coarser_windows_overlapping(adata_cell_filter_noGDF9, 'windowed_day_exp1', 'day_num', 'windowed_day_exp3')
    return adata_cell_filter_noGDF9


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into training and testing sets.")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to the directory containing input h5 files.")
    parser.add_argument("--meta", type=str, default=None, help="Path to metadata csv")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--kfolds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--out_file", type=str, default="wadot_processed.h5ad", help="Output file for the processed data.")
    parser.add_argument("--is_test", action='store_true', help="If set, run in test mode with smaller data.")
    
    args = parser.parse_args()

    wadot = load_h5(args.input_h5)

    if args.meta is not None:
        print(f'loading metadata from {args.meta}')
        meta = pd.read_csv(args.meta, index_col='cell')
        meta.columns.difference(wadot.obs.columns)
        tmp_join = pd.merge(wadot.obs, meta.loc[wadot.obs.index, meta.columns.difference(wadot.obs.columns)],
                            how='left', left_index=True, right_index=True)
        wadot.obs = tmp_join
    
    wadot = preprocess_data(wadot)
    wadot = in_dist_splits(wadot, args.kfolds, args.random_state)

    if args.is_test:
        print('test run on subset, keeping only 2000 cells and 500 hvgs')
        sc.pp.highly_variable_genes(wadot, n_top_genes=500, subset=True, flavor='seurat_v3')
        rand_idx = np.random.choice(wadot.n_obs, size=2000, replace=False)
        wadot = wadot[rand_idx, :].copy()
        

    # Out dist splits from time labels
    # All data does not require splits
    wadot.write_h5ad(args.out_file)

    output_file_base = args.out_file.replace('.h5ad', '')
    wadot_no_obs = wadot.copy()
    wadot_no_obs.obs = wadot_no_obs.obs.iloc[:, 0:0]  # keep index only
    wadot_no_obs.write(f"{output_file_base}_counts_only.h5ad")

    # 2. Save metadata to CSV
    wadot.obs.to_csv(f"{output_file_base}_obs_metadata.csv")
    wadot.var.to_csv(f"{output_file_base}_var_metadata.csv")