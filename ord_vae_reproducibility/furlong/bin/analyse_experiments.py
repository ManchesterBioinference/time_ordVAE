from pathlib import Path
import argparse
import json
import sys
import math
import itertools
from collections import OrderedDict

import pickle as pkl
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, entropy, spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import sklearn
import os

sys.path.append("/mnt/mr01-data01/j72687wm/")
import vae_models.utils
from vae_models.utils import string_to_splits_np, str_to_ord_levels

sys.path.append("/mnt/mr01-data01/j72687wm/ordVae_paper/scripts/")
from analyse_experiments_utils import *

# Add at top of file after imports
FONT_SIZE_TITLE = 8
FONT_SIZE_LABEL = 7
FONT_SIZE_TICK = 6
FONT_SIZE_LEGEND = 6
TEXT_WIDTH = 526


def add_res_to_df(sub_df, dataset, model, split, exp, experiment, skip_corr=False):
    row = {
        'dataset': dataset,
        'method': model,
        'split': split,
        'exp': exp
    }

    if (sub_df['pred_time'] == 0).all():
        print('pred time is all zero. Setting all metrics to default values for {}/{}/{}/{}.'.format(dataset, model, split, exp))
        row.update({'spearman': 0, 'kendall': 0, 'pearson': 0, 'mse': 100, 'accuracy': 0})

        print(sub_df.head())
        raise ValueError("All predicted times are zero.")
        return row
        
    elif (np.isnan(sub_df['pred_time']).any()):
        print('pred time contains nan. Setting all metrics to default values for {}/{}/{}/{}.'.format(dataset, model, split, exp))
        row.update({'spearman': 0, 'kendall': 0, 'pearson': 0, 'mse': 100, 'accuracy': 0})
        return row
    else:
        # Calculate correlations
        if skip_corr:
            sp, ke, pe = np.nan, np.nan, np.nan
        else:
            sp, ke, pe = corr(sub_df['pred_time'] , sub_df['gt_time_numeric'])

        row.update({'spearman': sp, 'kendall': ke, 'pearson': pe,
                    'mse': np.mean(np.square(sub_df['pred_time'] - sub_df['gt_time_numeric'])),
                    'accuracy': calculate_overlapping_accuracy(sub_df['gt_time_numeric'], sub_df['pred_class_overlapping'])})

        
        return row


def add_records_to_df(records, missing_counter, missing_records, experiment, split_col='split'):
    results = {}
    rows_list = [] # Collection list
    metadata_predictions = {}

    

    for rec in records:
        method = rec.get("method")
        dataset = rec.get("dataset")
        column = rec.get("column")  # sub experiment
        column_root = rec.get("column_root", column)  # get column_root if exists, else use column
        results_path = Path(rec.get("results"))

        print(f"Processing: method {method}, dataset {dataset}, column {column}")

        if not results_path.exists():
            print(f"[WARN] Missing results for {experiment}/{dataset}/{method}/{column}: {results_path}")
            # record missing
            key = (dataset, method, column_root)
            missing_counter[key] = missing_counter.get(key, 0) + 1
            missing_records.append({
                "experiment": experiment,
                "dataset": dataset,
                "method": method,
                "exp": column,
                "expected_path": str(results_path)
            })
            continue

        # Load results and do your analysis here
        mdf = pd.read_csv(results_path, index_col=0)

        metadata_predictions[method] = {}
        metadata_predictions[method]['has_nan'] = np.isnan(mdf['pred_time']).any()
        metadata_predictions[method]['n_nan'] = np.sum(np.isnan(mdf['pred_time']))
        metadata_predictions[method]['has_inf'] = np.isinf(mdf['pred_time']).any()
        metadata_predictions[method]['n_inf'] = np.sum(np.isinf(mdf['pred_time']))

        if metadata_predictions[method]['has_nan'] | metadata_predictions[method]['has_inf']:
            print(method, dataset, column_root)
            print(metadata_predictions[method])
            raise Exception('There are nans or infs in predictions.')

        if mdf['gt_window'].dtype == object:
            mdf['gt_time_numeric'] = mdf['gt_window'].map(WINDOW_MAP)
        elif mdf['gt_window'].dtype in [float, int]:
            mdf['gt_time_numeric'] = mdf['gt_window']
        else:
            print(f"Unknown dtype for gt_window: {mdf['gt_window'].dtype}")
            mdf['gt_time_numeric'] = mdf['gt_window']
        
        inv_map = {v: k for k, v in WINDOW_MAP.items()}

        if mdf['gt_window'].dtype == object:
            mdf['gt_time_class'] = mdf['gt_window']
        elif mdf['gt_window'].dtype in [float, int]:
            mdf['gt_time_class'] = mdf['gt_window'].map(inv_map)
        else:
            print(f"Unknown dtype for gt_window: {mdf['gt_window'].dtype}")
            mdf['gt_time_class'] = mdf['gt_window']


        gt_meta = {}
        gt_meta['has_nan'] = np.isnan(mdf['gt_time_numeric']).any()
        gt_meta['n_nan'] = np.sum(np.isnan(mdf['gt_time_numeric']))
        gt_meta['has_inf'] = np.isinf(mdf['gt_time_numeric']).any()
        gt_meta['n_inf'] = np.sum(np.isinf(mdf['gt_time_numeric']))

        if gt_meta['has_nan'] | gt_meta['has_inf']:
            print("GT", method, dataset, column_root)
            print(gt_meta)
            raise Exception("There are nans or infs in GT labels")        

        if dataset not in results:
            results[dataset] = {}
        if method not in results[dataset]:
            results[dataset][method] = {}

        results[dataset][method][column_root] = mdf
        
        if method in ['temporalVAE'] and not (results[dataset][method][column_root]['pred_time'] == 0).all():
            results[dataset][method][column_root]['y_pred_scaled'] = temp_vae_rescale_to_range(results[dataset][method][column_root]['pred_time'], 
                                                                    results[dataset][method][column_root]['gt_time_numeric'],
                                                                    ood=(experiment in ['out_dist']))
        else:
            results[dataset][method][column_root]['y_pred_scaled'] = rescale_to_range(results[dataset][method][column_root]['pred_time'],
                                                                                      results[dataset][method][column_root]['gt_time_numeric'],
                                                                                      ood=(experiment in ['out_dist']))

        results[dataset][method][column_root]['pred_time_unscaled'] = results[dataset][method][column_root]['pred_time']

        if method in ['pypsupertime', 'temporalVAE'] and not (results[dataset][method][column_root]['pred_time'] == 0).all():
            results[dataset][method][column_root]['pred_time'] = results[dataset][method][column_root]['y_pred_scaled']

        results[dataset][method][column_root]['pred_class_unique'] = find_windows_unique(UNIQUE_EDGES, UNIQUE_EDGES_STR , results[dataset][method][column_root]['pred_time'])

        results[dataset][method][column_root]['pred_class_overlapping'] = find_windows_overlapping(results[dataset][method][column_root]['pred_time'])

        if split_col is None:
            row_data = add_res_to_df(results[dataset][method][column_root], dataset, method, np.nan, column_root, experiment)
            rows_list.append(row_data)
        else:
            splits = results[dataset][method][column_root]['split'].unique()
            skip_corr = True if (experiment in ['out_dist']) else False
            for split in splits:

                sub_df = results[dataset][method][column_root][results[dataset][method][column_root]['split'] == split]

                row_data = add_res_to_df(sub_df, dataset, method, split, column_root, experiment, skip_corr=skip_corr)
                rows_list.append(row_data)

    new_df = pd.DataFrame(rows_list)
    new_df.set_index(['dataset', 'method', 'split', 'exp'], inplace=True)
        # Set the MultiIndex here

    return new_df, results


def find_windows_overlapping(predictions):
    """
    Maps numeric predictions to all overlapping windows they fall into.
    """
    mapped_results = []
    n_nomatch = 0

    for pred in predictions:
        matches = []
        for window in WINDOW_MAP.keys():
            start, end = window.strip().split('_')
            start = float(start)
            end = float(end)
            # Check if prediction falls within the window bounds
            if start <= pred <= end:
                matches.append(str(WINDOW_MAP[window]))
        
        if not matches:
            n_nomatch += 1
            mapped_results.append(np.nan)
        else:
            # Join multiple matches with a semicolon
            mapped_results.append(';'.join(matches))

    print(f'overlapping: missing matches: {n_nomatch}')
    return mapped_results


def plot_corr(corr_df, corr_type, method, intron_extron='extron', xlab='Bulk', ylab='Pseudo-bulk'):
    corr_df = corr_df.astype(float)
    fig, ax = plt.subplots(tight_layout=True,figsize=((0.25*TEXT_WIDTH/72.27), (0.3*TEXT_WIDTH)/72.27))

    sns.heatmap(corr_df, annot=False, xticklabels=True, yticklabels=True, ax=ax)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f'{method}')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0) 

    fig.savefig(f'corr_plots/{method}_{intron_extron}_{corr_type}.pdf',  bbox_inches='tight')


def diagonal_concentration(corr_df):
    diag = np.diag(corr_df.values.astype(float))
    off_diag = corr_df.values.astype(float)[~np.eye(len(corr_df), dtype=bool)]
    concentration = diag.mean() - off_diag.mean()
    return concentration


def calculate_corr_with_bulk(results, adata, methods, methods_names_short=None):
    #transcript_reads = pd.read_csv('/mnt/mr01-data01/j72687wm/ordVae_paper/datasets/bulk/drosophila_dev_transcripts.csv', index_col=0)
    id_to_gene = pd.read_csv('/mnt/mr01-data01/j72687wm/ordVae_paper/datasets/bulk/id_to_transcript.tsv',
                             header=4, index_col=0, sep='\t', skipfooter=1, usecols=[0, 1, 2, 3])
    genes_with_lengths = pd.read_csv('/mnt/mr01-data01/j72687wm/ordVae_paper/datasets/bulk/drosophila_dev_gene_reads_with_lengths.csv', index_col=0)

    os.mkdir('corr_plots')
    

    elife_nc_windows = [
        ('11', (95, 105)),
        ('12', (105, 115)),
        ('13A', (115, 125)),
        ('13B', (125, 145)),
        ('14A', (145, 160)),
        ('14B', (160, 175)),
        ('14C', (175, 190)),
        ('14D', (190, 205)),
        ('14E', (205, 220)),
        ('CF', (220, 235))
    ]
    
    if method_names_short is not None:
        results = {methods_names_short.get(method, method): results[method] for method in methods}
    methods = list(results.keys())

    gene_symbol = list(id_to_gene['gene_symbol'].astype('str'))
    upper_gene_symb = list(map(str.upper, gene_symbol))
    adata_genes_to_keep_names = [el.upper() for el in adata.var.index if el.upper() in upper_gene_symb]

    adata_genes_to_keep_names = [el.upper() for el in adata.var.index if el.upper() in upper_gene_symb]
    adata_genes_to_keep_bool = [el.upper() in upper_gene_symb for el in adata.var.index]
    elife_genes_to_keep_names = genes_with_lengths.index[genes_with_lengths.index.isin(id_to_gene.index)]
    elife_genes_to_keep_bool = genes_with_lengths.index.isin(id_to_gene.index)

    id_to_gene["gene_symbol_upper"] = id_to_gene["gene_symbol"].str.upper()
    genes_to_keep = id_to_gene.loc[(id_to_gene.index.isin(elife_genes_to_keep_names) \
                                      & (id_to_gene.gene_symbol_upper.isin(adata_genes_to_keep_names))),:]

    adata.var.index = adata.var.index.str.upper()
    
    # Make adata.var.index unique by keeping only first occurrence of duplicates
    adata = adata[:, ~adata.var.index.duplicated(keep='first')]

    # Get unique gene symbols to avoid duplicate index errors
    genes_to_keep_unique = genes_to_keep['gene_symbol_upper'].drop_duplicates().tolist()
    
    # Filter to only genes that exist in adata after deduplication
    genes_to_keep_unique = [g for g in genes_to_keep_unique if g in adata.var.index]

    psuedo_bulk_mean = {method: pd.DataFrame(index = ['11', '12', '13A', '13B', '14A', '14B', '14C', '14D', '14E', 'CF'],
                                               columns = genes_to_keep_unique) for method in methods}
    psuedo_bulk_rpkm = {method: pd.DataFrame(index = ['11', '12', '13A', '13B', '14A', '14B', '14C', '14D', '14E', 'CF'],
                                               columns = genes_to_keep_unique) for method in methods}
    # psuedo_bulk_median = {method: pd.DataFrame(index = ['11', '12', '13A', '13B', '14A', '14B', '14C', '14D', '14E', 'CF'],
    #                                          columns = genes_to_keep_unique) for method in methods}
    
    fb_to_symbol = genes_to_keep['gene_symbol_upper'].to_dict()
    adata_sub = adata[:, genes_to_keep_unique].copy()
    gene_lengths_exon = genes_with_lengths.loc[genes_to_keep.index, 'exon_size'].copy()
    gene_lengths_exon.index = gene_lengths_exon.index.map(fb_to_symbol)
    gene_lengths_exon = gene_lengths_exon[~gene_lengths_exon.index.duplicated(keep='first')]
    gene_lengths_exon = gene_lengths_exon.loc[genes_to_keep_unique]
    exon_lengths_kb = gene_lengths_exon.values / 1000.0 

    print(f"Exon lengths (kb) - min: {exon_lengths_kb.min():.2f}, max: {exon_lengths_kb.max():.2f}, mean: {exon_lengths_kb.mean():.2f}")

    print(adata_sub)

    for method in methods:
        print(method)
        for nc in elife_nc_windows:
            print(nc[1][0], nc[1][1])
            results[method]['pred_time_mins'] = results[method]['pred_time'] * 60
            row_mask = (results[method]['pred_time_mins'] >= nc[1][0]) & (results[method]['pred_time_mins'] < nc[1][1])
            idx_keep = results[method].index[row_mask]
            tmp_adata_sub = adata_sub[idx_keep, :].copy()
            
            subset_reads_mat = tmp_adata_sub.X.todense()
            
            grouped_reads_mean = np.mean(subset_reads_mat, axis=0)
            psuedo_bulk_mean[method].loc[nc[0], genes_to_keep_unique] = grouped_reads_mean

            # RPKM: sum counts per gene, normalise by library size (millions) and gene length (kb)
            # RPKM = (counts / exon_length_kb) / (total_counts / 1e6)
            grouped_reads_sum = subset_reads_mat.sum(axis=0)  # shape: (n_genes,)
            library_size_millions = subset_reads_mat.sum() / 1e6  # scalar
            if library_size_millions > 0:
                grouped_reads_rpkm = grouped_reads_sum / (exon_lengths_kb * library_size_millions)
            else:
                print(f"Warning: empty window {nc[0]} for method {method}, setting RPKM to zero")
                grouped_reads_rpkm = np.zeros_like(grouped_reads_sum, dtype=float)
            psuedo_bulk_rpkm[method].loc[nc[0], genes_to_keep_unique] = grouped_reads_rpkm

            del subset_reads_mat, tmp_adata_sub

        psuedo_bulk_mean[method] = psuedo_bulk_mean[method].astype(float)
        psuedo_bulk_rpkm[method] = psuedo_bulk_rpkm[method].astype(float)

    genes_sub = genes_with_lengths.loc[genes_to_keep.index,genes_with_lengths.columns[:-2]].T.astype(float)
    
    # Convert FB gene IDs (columns) to gene symbols using id_to_gene mapping
    
    genes_sub.columns = genes_sub.columns.map(fb_to_symbol)
    genes_sub = genes_sub.loc[:, genes_to_keep_unique].copy()
    
    # Take median across replicates (indicated by 1, 2, or 3 in the second position: *_1_*, *_2_*, *_3_*)
    # Extract the base name by removing the replicate identifier
    genes_sub_col_groups = genes_sub.index.str.replace(r'_[123]_', '_', regex=True)
    genes_sub = genes_sub.groupby(genes_sub_col_groups).median()
    genes_sub.index = genes_sub.index.str.split('_').str[0]

    genes_sub_exon = genes_sub.loc[[el for el in genes_sub.index if "I" not in el], :]
    genes_sub_intron = genes_sub.loc[[el for el in genes_sub.index if "I" in el], :]

    genes_sub_exon.index = genes_sub_exon.index.str[1:]
    genes_sub_intron.index = genes_sub_intron.index.str[1:]

    if not (psuedo_bulk_mean[method].index == genes_sub_exon.index).all():
        print(f"Index mismatch for method {method}")
        raise Exception("Index mismatch between pseudo bulk and genes_sub_exon")

    if not (psuedo_bulk_mean[method].columns == genes_sub_exon.columns).all():
        print(f"Column mismatch for method {method}")
        raise Exception("Column mismatch between pseudo bulk and genes_sub_exon")

    pear_mat_mean_exon = {method: pd.DataFrame(index = genes_sub_exon.index,
                                               columns = psuedo_bulk_mean[method].index) for method in methods}
    spear_mat_mean_exon = {method: pd.DataFrame(index = genes_sub_exon.index,
                                                columns = psuedo_bulk_mean[method].index)for method in methods}
    pear_mat_rpkm_exon = {method: pd.DataFrame(index = genes_sub_exon.index,
                                               columns = psuedo_bulk_rpkm[method].index) for method in methods}
    spear_mat_rpkm_exon = {method: pd.DataFrame(index = genes_sub_exon.index,
                                                columns = psuedo_bulk_rpkm[method].index) for method in methods}

    for method in methods:
        for i, j in itertools.product(genes_sub_exon.index, psuedo_bulk_mean[method].index):

            pear_mat_mean_exon[method].loc[i, j] = pearsonr(genes_sub_exon.loc[i, :].values,
                                                            psuedo_bulk_mean[method].loc[j, :].values)[0]
            spear_mat_mean_exon[method].loc[i, j] = spearmanr(genes_sub_exon.loc[i, :].values,
                                                           psuedo_bulk_mean[method].loc[j, :].values)[0]
            pear_mat_rpkm_exon[method].loc[i, j] = pearsonr(genes_sub_exon.loc[i, :].values,
                                                            psuedo_bulk_rpkm[method].loc[j, :].values)[0]
            spear_mat_rpkm_exon[method].loc[i, j] = spearmanr(genes_sub_exon.loc[i, :].values,
                                                              psuedo_bulk_rpkm[method].loc[j, :].values)[0]

            

    for method in methods:
        for corr_type, df in [('pearson', pear_mat_mean_exon[method]), ('spearman', spear_mat_mean_exon[method])]:
            plot_corr(df, corr_type, method, intron_extron='exon_mean', xlab='Bulk', ylab='Pseudo-bulk')
        for corr_type, df in [('pearson', pear_mat_rpkm_exon[method]), ('spearman', spear_mat_rpkm_exon[method])]:
            plot_corr(df, corr_type, method, intron_extron='exon_rpkm', xlab='Bulk', ylab='Pseudo-bulk')

    # Get self corr as baseline
    pear_mat_bulk_self = pd.DataFrame(index=genes_sub_exon.index, columns=genes_sub_exon.index)
    spear_mat_bulk_self = pd.DataFrame(index=genes_sub_exon.index, columns=genes_sub_exon.index)

    for i, j in itertools.product(genes_sub_exon.index, genes_sub_exon.index):
        pear_mat_bulk_self.loc[i, j] = pearsonr(genes_sub_exon.loc[i, :].values,
                                                 genes_sub_exon.loc[j, :].values)[0]
        spear_mat_bulk_self.loc[i, j] = spearmanr(genes_sub_exon.loc[i, :].values,
                                                   genes_sub_exon.loc[j, :].values)[0]

    plot_corr(pear_mat_bulk_self, 'pearson', 'bulk_self', intron_extron='exon', xlab='Bulk', ylab='Bulk')
    plot_corr(spear_mat_bulk_self, 'spearman', 'bulk_self', intron_extron='exon', xlab='Bulk', ylab='Bulk')

    # Get boxplot of the on-diagnoal correlation scores
    method_corr_on_diag_pear = {'mean': {}, 'rpkm': {}}
    method_corr_on_diag_spear = {'mean': {}, 'rpkm': {}}

    for method in methods:
        method_corr_on_diag_pear['mean'][method] = [float(pear_mat_mean_exon[method].loc[nc[0], nc[0]]) for nc in elife_nc_windows]
        method_corr_on_diag_spear['mean'][method] = [float(spear_mat_mean_exon[method].loc[nc[0], nc[0]]) for nc in elife_nc_windows]
        method_corr_on_diag_pear['rpkm'][method] = [float(pear_mat_rpkm_exon[method].loc[nc[0], nc[0]]) for nc in elife_nc_windows]
        method_corr_on_diag_spear['rpkm'][method] = [float(spear_mat_rpkm_exon[method].loc[nc[0], nc[0]]) for nc in elife_nc_windows]

    for agg_type in ['mean', 'rpkm']:
        for corr_type, diag_dict in [('pearson', method_corr_on_diag_pear[agg_type]),
                                      ('spearman', method_corr_on_diag_spear[agg_type])]:
            labels, data = diag_dict.keys(), diag_dict.values()
            fig, ax = plt.subplots(tight_layout=True, figsize=((0.25*TEXT_WIDTH/72.27), (0.3*(TEXT_WIDTH)/72.27)))
            ax.boxplot(data)
            ax.set_xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
            ax.set_title(f'{corr_type} on-diagonal ({agg_type})')
            fig.savefig(f'corr_plots/{corr_type}_on_diag_{agg_type}_boxplot.pdf', bbox_inches='tight')
            plt.close(fig)

    concentration_scores = {
        'mean': {'pearson': {}, 'spearman': {}},
        'rpkm': {'pearson': {}, 'spearman': {}}
    }

    corr_mats = {
        'mean': {
            'pearson': pear_mat_mean_exon,
            'spearman': spear_mat_mean_exon
        },
        'rpkm': {
            'pearson': pear_mat_rpkm_exon,
            'spearman': spear_mat_rpkm_exon
        }
    }

    for agg_type in ['mean', 'rpkm']:
        for corr_type in ['pearson', 'spearman']:
            for method in methods:
                concentration = diagonal_concentration(corr_mats[agg_type][corr_type][method])
                concentration_scores[agg_type][corr_type][method] = concentration
                print(f"{method} {agg_type} {corr_type} diagonal concentration: {concentration:.3f}")

    # Use shared y-limits across all plots for easier comparison
    all_concentrations = []
    for agg_type in ['mean', 'rpkm']:
        for corr_type in ['pearson', 'spearman']:
            all_concentrations.extend(concentration_scores[agg_type][corr_type].values())

    y_min = min(all_concentrations)
    y_max = max(all_concentrations)
    y_pad = max(0.02, 0.05 * (y_max - y_min if y_max > y_min else 1.0))

    for agg_type in ['mean', 'rpkm']:
        for corr_type in ['pearson', 'spearman']:
            fig, ax = plt.subplots(
                tight_layout=True,
                figsize=((0.25 * TEXT_WIDTH / 72.27), (0.3 * (TEXT_WIDTH) / 72.27))
            )

            labels = list(methods)
            vals = [concentration_scores[agg_type][corr_type][method] for method in labels]

            ax.bar(range(len(labels)), vals)
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
            ax.set_ylabel('Diagonal concentration')
            ax.set_title(f'{corr_type}, {agg_type}')
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

            fig.savefig(f'corr_plots/diagonal_concentration_{agg_type}_{corr_type}.pdf', bbox_inches='tight')
            plt.close(fig)

    os.mkdir('gene_plots')

    # Plot gene expression vs predicted time with Gaussian kernel smoothing
    gene_list = ['nos', 'Meltrin', 'upd1', 'wg', 'sna', 'twi', 'dl', 'cad', 'bcd', 'hb']
    gene_list_upper = list(map(str.upper, gene_list))


    # Per-gene scatter plots of pseudo-bulk vs bulk RPKM (across timepoints)
    i = 0
    for gene in gene_list_upper:
        if gene not in genes_sub_exon.columns:
            print(f"Gene {gene} not found in genes_sub_exon, skipping scatter plot...")
            continue
        if gene not in psuedo_bulk_rpkm[methods[0]].columns:
            print(f"Gene {gene} not found in pseudo_bulk_rpkm, skipping scatter plot...")
            continue
            
        # Get bulk values for this gene (across timepoints)
        bulk_gene_vals = genes_sub_exon.loc[:, gene].values
        
        # Plot all methods on one figure
        fig, ax = plt.subplots(tight_layout=True, figsize=((0.25*TEXT_WIDTH/72.27), (0.3*TEXT_WIDTH/72.27)))
        
        for method in methods:
            pseudo_bulk_gene_vals = psuedo_bulk_rpkm[method].loc[:, gene].values
            ax.scatter(bulk_gene_vals, pseudo_bulk_gene_vals, alpha=0.7, s=15, label=method)
        
        # Add diagonal line for reference
        all_vals = np.concatenate([bulk_gene_vals] + [psuedo_bulk_rpkm[m].loc[:, gene].values for m in methods])
        max_val = np.nanmax(all_vals)
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.8, alpha=0.5)
        
        ax.set_title(f'{gene}')
        ax.set_ylabel('Bulk (RPKM)')
        ax.set_xlabel('Pseudo-bulk (RPKM)')
        fig.savefig(f'gene_plots/scatter_bulk_vs_pseudobulk_{gene}.pdf', bbox_inches='tight')
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
        if i == 0:
            fig.savefig(f'gene_plots/scatter_bulk_vs_pseudobulk_legend.pdf', bbox_inches='tight')
            i=1
        
        plt.close(fig)


    for gene in gene_list_upper:
        if gene not in adata_sub.var.index:
            print(f"Gene {gene} not found in adata, skipping...")
            continue
        if gene not in genes_sub_exon.columns:
            print(f"Gene {gene} not found in genes_sub_exon, skipping...")
            continue
            
        # Get expression values for this gene
        gene_expr = np.array(adata_sub[:, gene].X.todense()).flatten()
        bulk_expression = genes_sub_exon.loc[:, gene].values        
        bulk_times = []
        for stage in genes_sub_exon.index:
            bulk_times.append( next(nc_wind[1][0] for nc_wind in elife_nc_windows if stage == nc_wind[0]))
        
        for method in methods:
            fig, ax = plt.subplots(tight_layout=True, figsize=((0.25*TEXT_WIDTH/72.27), (0.3*TEXT_WIDTH/72.27)))
            pred_time = results[method]['pred_time_mins'].values
            
            # Create time grid for smoothing
            
            t_grid = np.linspace(80, 250, 1000)
            
            # Gaussian kernel smoothing parameters
            bandwidth = (250 - 80) / 20  # bandwidth as 5% of time range
            
            # Compute Gaussian-weighted average at each grid point
            smoothed_expr = np.zeros_like(t_grid)
            for k, t in enumerate(t_grid):
                weights = np.exp(-0.5 * ((pred_time - t) / bandwidth) ** 2)
                weights /= weights.sum()
                smoothed_expr[k] = np.sum(weights * gene_expr)

            # Min-max scale both curves to [0, 1] for shape comparison
            smoothed_min, smoothed_max = smoothed_expr.min(), smoothed_expr.max()
            if smoothed_max > smoothed_min:
                smoothed_expr_scaled = (smoothed_expr - smoothed_min) / (smoothed_max - smoothed_min)
            else:
                smoothed_expr_scaled = np.zeros_like(smoothed_expr)
            
            bulk_min, bulk_max = bulk_expression.min(), bulk_expression.max()
            if bulk_max > bulk_min:
                bulk_expression_scaled = (bulk_expression - bulk_min) / (bulk_max - bulk_min)
            else:
                bulk_expression_scaled = np.zeros_like(bulk_expression)
            
            ax.plot(t_grid, smoothed_expr_scaled, label=method, linewidth=1.5)

            ax.plot(bulk_times, bulk_expression_scaled, label='Bulk', marker='o', linestyle='--', color='black', markersize=4)
        
            ax.set_title(f'{gene}')
            ax.set_xlabel('Predicted time')
            ax.set_ylabel('Expression (smoothed)')
            ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
            fig.savefig(f'gene_plots/gene_{gene}_{method}_expr_vs_time_smoothed.pdf', bbox_inches='tight')
            plt.close(fig)


def time_hists(results):

    os.mkdir('time_hists')

    methods = ['sceptic', 'temporalVAE', 'nb_cumulative_logit']
    method_short = {'sceptic': 'SCEPTIC', 'temporalVAE': 'TempVAE', 'nb_cumulative_logit': 'nb_cl'}

    times = ['14_18', '16_20']

    fig, axs = plt.subplots(1, 3, figsize=((0.75*TEXT_WIDTH)/72.27,  (0.75*0.618*0.5*TEXT_WIDTH) / 72.27), tight_layout=True)
    for ax, method in zip(axs, methods):
        for time in times:
            df = results[method]
            pred_time = df[df['gt_time_class'] == time]['pred_time'].values

            bin_counts, bin_edges = np.histogram(pred_time, bins=100, density=True)
            line = ax.stairs(bin_counts, bin_edges, alpha=0.8, label=time)
            c = line.get_edgecolor()
            if np.ndim(c) > 1:
                c = c[0]  # some matplotlib versions return shape (1, 4)

            ax.axvline(time, linestyle='--', color=c)

        ax.set_title(f'{method_short[method]}')
        ax.set_xlabel('Predicted time')
        ax.set_ylabel('Density')
        ax.set_xlim(12, 21)
        ax.legend()
    fig.savefig(f'time_hists/{method}_hist_14_20.pdf', bbox_inches='tight')
    plt.close()

    times = ['00_02', '01_03']

    fig, axs = plt.subplots(1, 3, figsize=((0.75*TEXT_WIDTH)/72.27,  (0.75*0.618*0.5*TEXT_WIDTH) / 72.27), tight_layout=True)
    for ax, method in zip(axs, methods):
        for time in times:
            df = results[method]
            pred_time = df[df['gt_time_class'] == time]['pred_time'].values

            bin_counts, bin_edges = np.histogram(pred_time, bins=100, density=True)
            line = ax.stairs(bin_counts, bin_edges, alpha=0.8, label=time)
            c = line.get_edgecolor()
            if np.ndim(c) > 1:
                c = c[0]  # some matplotlib versions return shape (1, 4)

            ax.axvline(time, linestyle='--', color=c)

        ax.set_title(f'{method_short[method]}')
        ax.set_xlabel('Predicted time')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 5)
        ax.legend()
    fig.savefig(f'time_hists/{method}_hist_0_3.pdf', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment metrics")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--records", required=True, help="path to YAML file with records")
    parser.add_argument("--metrics", required=False, help="Comma-separated list of metrics to plot", default="mse,accuracy")
    parser.add_argument("--method_colours", required=False, help="JSON string of method color indices", default=None)
    parser.add_argument("--colormap", required=False, help="Matplotlib colormap name", default='tab20')
    parser.add_argument("--publishdir", required=False, help="Directory to save publication-ready plots", default=None)
    args = parser.parse_args()

    # Parse method colors if provided
    method_colours_indices = {}
    if args.method_colours:
        try:
            method_colours_indices = json.loads(args.method_colours)
            print(f"Parsed method_colours: {method_colours_indices}")
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse method_colours JSON: {e}")
            print(f"Using empty dictionary instead")
    
    
    set_publication_style()

    missing_counter = {}
    missing_records = []  # add

    experiment = args.experiment
    metrics_str = args.metrics
    if metrics_str:
        metrics = metrics_str.split(',')

    # Read JSON file
    records_path = Path(args.records)
    if not records_path.exists():
        print(f"Error: records file not found: {records_path}", file=sys.stderr)
        sys.exit(2)
    try:
        records = json.loads(records_path.read_text())
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {records_path}: {e}", file=sys.stderr)
        sys.exit(3)

    anndata = sc.read_h5ad('/mnt/mr01-data01/j72687wm/ordVae_paper/nextflow_furlong/results/data/furlong_processed.h5ad')

    mapping_df = anndata.obs[['time_windows', 'time_windows_mean']].drop_duplicates()
    WINDOW_MAP = dict(zip(mapping_df['time_windows'], mapping_df['time_windows_mean']))
    _, UNIQUE_EDGES, _ = string_to_splits_np(anndata.obs['time_windows'], inf_edge=True)
    UNIQUE_EDGES_STR = ['_'.join(window) for window in UNIQUE_EDGES.astype(str)]

    pubdir = args.publishdir
    if (pubdir is not None) and os.path.isfile(os.path.join(pubdir, 'results.pkl')):
        print(f"Loading existing results from {pubdir}/results.pkl")
        with open(os.path.join(pubdir, 'results.pkl'), 'rb') as f:
            df, results = pkl.load(f)
        
    else:
        df, results = add_records_to_df(records, missing_counter, missing_records, experiment)
        if pubdir is not None:
            with open(os.path.join(pubdir, 'results.pkl'), 'wb') as f:
                pkl.dump((df, results), f)

    output_path = Path(f"combined_metrics_{experiment}.csv")
    df.to_csv(output_path)
    print(f"Analysis results saved to {output_path}")

    # Generate plots
    #plot spearman by exp

    # use named level for clarity
    if 'method' in df.index.names:
        methods = df.index.get_level_values('method').unique().tolist()
    else:
        methods = df.index.get_level_values(1).unique().tolist()

    method_names_short = {
        'sceptic': 'sceptic',
        'ga_reg':'ga_reg',
        'ga_ord':'ga_ord',
        'ga_cumulative_logit':'ga_cl',
        'nb_reg':'nb_reg',
        'nb_ord':'nb_ord',
        'nb_cumulative_logit':'nb_cl',
        'pypsupertime':'psupertime',
        'temporalVAE':'tempVAE'
    }

    metric_fig_size = (0.33*TEXT_WIDTH / 72.27, (0.618*0.5*TEXT_WIDTH) / 72.27)
    for metric in metrics:
        plot_boxplot_metric_by_sub_exp(df, experiment, methods, metric=metric, title=f"{metric}_by_col_{experiment}.pdf",
                                       method_names_short=method_names_short, figsize=metric_fig_size)
    
    rank_fig_size = (0.25*TEXT_WIDTH / 72.27, (0.75*0.618*0.5*TEXT_WIDTH) / 72.27)
    plot_rank_by_column_root(df, experiment, metrics, 
                            method_color_indices=method_colours_indices,
                            colormap=args.colormap,title=f"rank_by_exp_{experiment}.pdf", figsize=rank_fig_size,
                            method_names_short=method_names_short)

        # After processing all records, write missing reports
    if missing_records:
        df_missing = pd.DataFrame(missing_records)
        df_missing.to_csv(f"missing_files_{experiment}.csv", index=False)

        summary = (
            df_missing.groupby(["dataset", "method", "exp"])
            .size()
            .reset_index(name="missing_count")
            .sort_values(["missing_count", "dataset", "method", "exp"], ascending=[False, True, True, True])
        )
        summary.to_csv(f"missing_summary_{experiment}.csv", index=False)
        print(f"Wrote {len(df_missing)} missing entries to missing_files_{experiment}.csv and summary to missing_summary_{experiment}.csv")
    else:
        print("No missing result files detected.")

    print("generating plots without splits for out_dist experiment...")
    missing_counter_ns = {}
    missing_records_ns = []  # add

    results_trim = {}

    for dataset in results:
        print(dataset)
        for method in results[dataset]:
            print(method)
            for column_root in results[dataset][method]:
                print(column_root)
                results_trim[method] = results[dataset][method][column_root]

    os.mkdir('psuedo_time_dist_plots')

    for method in results_trim:
        plot_psuedo_time_dist(results_trim[method], method, save_loc=f'psuedo_time_dist_plots/pseudo_time_dist_{method}.pdf')

    calculate_corr_with_bulk(results_trim, anndata, methods, methods_names_short=method_names_short)
    time_hists(results_trim)

    if experiment in ['out_dist']:
        if (pubdir is not None) and os.path.isfile(os.path.join(pubdir, 'results_no_split.pkl')):
            print(f"Loading existing results from {pubdir}/results_no_split.pkl")
            with open(os.path.join(pubdir, 'results_no_split.pkl'), 'rb') as f:
                df_no_split, results_no_split = pkl.load(f)
            
        else:
            df_no_split, results_no_split = add_records_to_df(records, missing_counter_ns, missing_records_ns, experiment, split_col=None)
            if pubdir is not None:
                with open(os.path.join(pubdir, 'results_no_split.pkl'), 'wb') as f:
                    pkl.dump((df_no_split, results_no_split), f)
        

        # use named level for clarity
        if 'method' in df_no_split.index.names:
            methods = df_no_split.index.get_level_values('method').unique().tolist()
        else:
            methods = df_no_split.index.get_level_values(1).unique().tolist()

        metrics = ['spearman', 'kendall', 'pearson', 'mse', 'accuracy']

        print(df_no_split.head())
        for metric in metrics:
            plot_boxplot_metric_by_sub_exp(df_no_split, experiment, methods, metric=metric, title=f"{metric}_by_col_{experiment}_no_splits.pdf",
                                           method_names_short=method_names_short, figsize=metric_fig_size)
        

        plot_rank_by_column_root(df_no_split, experiment, metrics,
                                 method_color_indices=method_colours_indices,
                                 colormap=args.colormap,
                                 title=f"rank_by_col_{experiment}_no_splits.pdf", figsize=rank_fig_size,
                                 method_names_short=method_names_short)

        print("generating plots based on extrapolation and interpolation...")
        split_values = df.index.get_level_values('split').unique().dropna()            
        # Ensure we treat them as floats for comparison
        split_floats = split_values.astype(float)
        min_split = split_floats.min()
        max_split = split_floats.max()
        
        print(f"Timepoints detected - Min: {min_split}, Max: {max_split}")

        # 2. Create Boolean masks using the MultiIndex values
        idx_vals = df.index.get_level_values('split').astype(float)
        
        is_extrap = (idx_vals == min_split) | (idx_vals == max_split)
        is_interp = ~is_extrap

        # 3. Subset DataFrame
        df_extrap = df[is_extrap]
        df_interp = df[is_interp]

        metrics = ['mse', 'accuracy']

        # 4. Plot Extrapolation
        if not df_extrap.empty:
            print(f"Plotting Extrapolation ({len(df_extrap)} rows)")
            for metric in metrics:
                plot_boxplot_metric_by_sub_exp(df_extrap, experiment, methods, metric=metric, figsize=metric_fig_size,
                                            title=f"{metric}_by_col_{experiment}_extrapolation.pdf",
                                            method_names_short=method_names_short)
            plot_rank_by_column_root(df_extrap, experiment, metrics,
                                   method_color_indices=method_colours_indices,
                                   colormap=args.colormap,
                                   title=f"rank_by_col_{experiment}_extrapolation.pdf", figsize=rank_fig_size,
                                   method_names_short=method_names_short)

        metrics = ['mse', 'accuracy']

        # 5. Plot Interpolation
        if not df_interp.empty:
            print(f"Plotting Interpolation ({len(df_interp)} rows)")
            for metric in metrics:
                plot_boxplot_metric_by_sub_exp(df_interp, experiment, methods, metric=metric, figsize=metric_fig_size,
                                            title=f"{metric}_by_col_{experiment}_interpolation.pdf",
                                            method_names_short=method_names_short)
            plot_rank_by_column_root(df_interp, experiment, metrics,
                                   method_color_indices=method_colours_indices,
                                   colormap=args.colormap, 
                                   title=f"rank_by_col_{experiment}_interpolation.pdf", figsize=rank_fig_size,
                                   method_names_short=method_names_short)   