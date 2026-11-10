import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import median_abs_deviation
import argparse
import os

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

def preprocess_data(adata):
    adata.var["mt"] = adata.var_names.str.startswith("mt:")
    sc.pp.calculate_qc_metrics(furlong, qc_vars=["mt"], inplace=True, percent_top=[20], log1p=True)

    adata.obs["outlier"] = (
    is_outlier_by_batch(adata, "log1p_total_counts", 5, batch_col='exp')
    | is_outlier_by_batch(adata, "log1p_n_genes_by_counts", 5, batch_col='exp')
    | is_outlier_by_batch(adata, "pct_counts_in_top_20_genes", 5, batch_col='exp')
    )

    adata.obs["mt_outlier"] = is_outlier_by_batch(adata, "pct_counts_mt", 3, batch_col='exp') | (
      adata.obs["pct_counts_mt"] > 8
    )
    adata_cell_filter = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()

    adata_cell_filter.obs['time_windows'] = adata_cell_filter.obs['time'].str.split('_').str[1:].str.join('_')
    adata_cell_filter.obs['time_windows_mean'] = adata_cell_filter.obs['time_windows'].str.split('_').apply(lambda x: np.mean([float(x[0]), float(x[1])]))

    return adata_cell_filter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into training and testing sets.")
    parser.add_argument("--input_h5ad", type=str, required=True, help="Path to the input h5ad file.")
    parser.add_argument("--meta", type=str, default=None, help="Path to save the training h5ad file.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--kfolds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--out_file", type=str, default="furlong_processed.h5ad", help="Output file for the processed data.")
    
    args = parser.parse_args()

    furlong = sc.read_h5ad(args.input_h5ad)

    if args.meta is not None:
        meta = pd.read_csv(args.meta, index_col='cell')
        meta.columns.difference(furlong.obs.columns)
        tmp_join = pd.merge(furlong.obs, meta.loc[furlong.obs.index, meta.columns.difference(furlong.obs.columns)],
                            how='left', left_index=True, right_index=True)
        furlong.obs = tmp_join

    furlong = preprocess_data(furlong)
    in_dist_splits(furlong, args.kfolds, args.random_state)


    # Out dist splits from time labels
    # All data does not require splits
    furlong.write_h5ad(args.out_file)

    output_file_base = args.out_file.replace('.h5ad', '')
    furlong_no_obs = furlong.copy()
    furlong_no_obs.obs = furlong_no_obs.obs.iloc[:, 0:0]  # keep index only
    furlong_no_obs.write(f"{output_file_base}_counts_only.h5ad")

    # 2. Save metadata to CSV
    furlong.obs.to_csv(f"{output_file_base}_obs_metadata.csv")
    furlong.var.to_csv(f"{output_file_base}_var_metadata.csv")