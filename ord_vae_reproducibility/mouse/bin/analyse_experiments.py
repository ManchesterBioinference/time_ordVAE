from pathlib import Path
import argparse
import json
import sys
import math
import itertools
from collections import OrderedDict
import os

import pickle as pkl
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, entropy, spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import sklearn
from sklearn.metrics import accuracy_score

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


def find_windows_overlapping(predictions, column):
    """
    Maps numeric predictions to all overlapping windows they fall into.
    """
    mapped_results = []
    n_nomatch = 0

    for pred in predictions:
        matches = []
        for window in WINDOW_MAP_PER_COL[column].keys():
            start, end = window.strip().split('_')
            start = float(start)
            end = float(end)
            # Check if prediction falls within the window bounds
            if start <= pred <= end:
                matches.append(str(WINDOW_MAP_PER_COL[column][window]))
        
        if not matches:
            n_nomatch += 1
            mapped_results.append(np.nan)
        else:
            # Join multiple matches with a semicolon
            mapped_results.append(';'.join(matches))

    print(f'overlapping: missing matches: {n_nomatch}')
    return mapped_results


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
    elif (np.isnan(sub_df['gt_time_numeric']).any()):
        print('gt_time_numeric contains nan. Setting all metrics to default values for {}/{}/{}/{}.'.format(dataset, model, split, exp))
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
                    'accuracy_train_windows': calculate_overlapping_accuracy(sub_df['gt_time_numeric'], sub_df['pred_class_overlapping_train_windows']),
                    'accuracy_gt_windows': accuracy_score(sub_df['gt_time_numeric'].astype(str), sub_df['pred_class_overlapping_gt_windows'].astype(str))})
        
        return row


def add_records_to_df(records, missing_counter, missing_records, experiment, split_col='split', gt_window_series=None, filter_split=None, idx_filter=None):
    """
    Args:
        filter_split: If provided, only include rows where split column matches this value.
                      Can be 'train', 'test', or None (include all).
    """
    results = {}
    rows_list = [] # Collection list

    for rec in records:
        method = rec.get("method")
        dataset = rec.get("dataset")
        column = rec.get("column")
        column_suffix = rec.get("column")  # sub experiment
        results_path = Path(rec.get("results"))

        print(f"\n\n\n\nProcessing record: method={method}, dataset={dataset}, column={column}, results_path={results_path}\n")

        if not results_path.exists():
            print(f"[WARN] Missing results for {experiment}/{dataset}/{method}/{column}: {results_path}")
            # record missing
            key = (dataset, method, column)
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
        
        # Filter by train/test if specified
        if filter_split is not None and 'test_train' in mdf.columns:
            mdf = mdf[mdf['test_train'] == filter_split]
            if mdf.empty:
                print(f"[WARN] No {filter_split} data found for {dataset}/{method}/{column}")
                continue

        if idx_filter is not None:
            # Use .isin() to filter by index since idx_filter is an Index object, not a boolean mask
            original_len = len(mdf)
            mdf = mdf[mdf.index.isin(idx_filter)]
            print(f"DEBUG: Filtered from {original_len} to {len(mdf)} rows using idx_filter")
            # Check which day_num values remain after filtering
            remaining_days = anndata.obs.loc[mdf.index, 'day_num'].unique()
            print(f"DEBUG: day_num values after idx_filter: {remaining_days}")
            if mdf.empty:
                print(f"[WARN] No data left after index filtering for {dataset}/{method}/{column}")
                continue
        
        mdf['train_window'] = mdf['gt_window']
       
        if gt_window_series is not None:
            mdf['gt_window'] = gt_window_series[mdf.index]

        if (mdf['split'].dtype not in [int, float]) and (mdf['split'].iloc[0] != 'all_data'):
            mdf['split'] = mdf['split'].map(WINDOW_MAP_PER_COL[column])

        # Ensure each cell has a categorical and a numeric representation of the ground truth time
        if (mdf['gt_window'].dtype == object) | (mdf['gt_window'].dtype.name == 'category'):
            mdf['gt_time_numeric'] = mdf['gt_window'].map(WINDOW_MAP_PER_COL['windowed_day_base'])
            mdf['gt_time_numeric'] = pd.to_numeric(mdf['gt_time_numeric'])
            if mdf['gt_time_numeric'].isna().any():
                print(f"Warning: NaN values found in gt_time_numeric for {dataset}/{method}/{column}. Check WINDOW_MAP_PER_COL and gt_window values.")
                print(WINDOW_MAP_PER_COL['windowed_day_base'])
                print(mdf['gt_window'])
                print(mdf['gt_time_numeric'])
                raise ValueError("NaN values found in gt_time_numeric after mapping. Check the warning messages for details.")
            
        elif mdf['gt_window'].dtype in [float, int]:
            mdf['gt_time_numeric'] = mdf['gt_window']
        else:
            print(f"Unknown dtype for gt_window: {mdf['gt_window'].dtype}")
            mdf['gt_time_numeric'] = mdf['gt_window']
        
        # Create inverse mapping for gt_window to get categorical time class back from numeric
        inv_map = {v: k for k, v in WINDOW_MAP_PER_COL['windowed_day_base'].items()}

        if (mdf['gt_window'].dtype == object) | (mdf['gt_window'].dtype.name == 'category'):
            mdf['gt_time_class'] = mdf['gt_window']
        elif mdf['gt_window'].dtype in [float, int]:
            mdf['gt_time_class'] = mdf['gt_window'].map(inv_map)
        else:
            print(f"Unknown dtype for gt_window: {mdf['gt_window'].dtype}")
            mdf['gt_time_class'] = mdf['gt_window']

        # Ensure each cell has a categorical and a numeric representation of label used during training
        if (mdf['train_window'].dtype == object) | (mdf['train_window'].dtype.name == 'category'):
            mdf['train_time_numeric'] = mdf['train_window'].map(WINDOW_MAP_PER_COL[column])
        elif mdf['train_window'].dtype in [float, int]:
            mdf['train_time_numeric'] = mdf['train_window']
        else:
            print(f"Unknown dtype for train_window: {mdf['train_window'].dtype}")
            mdf['train_time_numeric'] = mdf['train_window']

        mdf['y_pred_scaled'] = temp_vae_rescale_to_range(mdf['pred_time'], 
                                                            mdf['gt_time_numeric'],
                                                            ood=(experiment in ['out_dist']))

        if method in ['temporalVAE'] and not (mdf['pred_time'] == 0).all():
            mdf['y_pred_scaled'] = temp_vae_rescale_to_range(mdf['pred_time'], 
                                                            mdf['gt_time_numeric'],
                                                            ood=(experiment in ['out_dist']))
        elif not (mdf['pred_time'] == 0).all():
            mdf['y_pred_scaled'] = rescale_to_range(mdf['pred_time'],
                                                    mdf['gt_time_numeric'],
                                                    ood=(experiment in ['out_dist']))
        else:
            print(f"Warning: All predicted times are zero for {dataset}/{method}/{column}. Skipping rescaling.")

        mdf['pred_time_unscaled'] = mdf['pred_time']

        if method in ['pypsupertime', 'temporalVAE'] and not (mdf['pred_time'] == 0).all():
            mdf['pred_time'] = mdf['y_pred_scaled']
        
        # Map predicted times to unique windows (unique edges per col created from potentially overlapping train windows)
        # This is not used later on
        mdf['pred_class_unique'] = find_windows_unique(UNIQUE_EDGES_PER_COL[column], UNIQUE_EDGES_STR_PER_COL[column],
                                                                                    mdf['pred_time'])
        
        # Map predicted times to potentially overlapping train windows
        mdf['pred_class_overlapping_train_windows'] = find_windows_overlapping(mdf['pred_time'], column)

        # Map predicted windows to ground truth windows (which are non-overlapping)
        mdf['pred_class_overlapping_gt_windows'] = find_windows_unique(UNIQUE_EDGES_PER_COL['windowed_day_base'],
                                                                                                    UNIQUE_EDGES_STR_PER_COL['windowed_day_base'],
                                                                                                    mdf['pred_time'])
        mdf['pred_class_overlapping_gt_windows']  = mdf['pred_class_overlapping_gt_windows'].map(WINDOW_MAP_PER_COL['windowed_day_base'])

        if dataset not in results:
            results[dataset] = {}
        if method not in results[dataset]:
            results[dataset][method] = {}

        results[dataset][method][column] = mdf

        print(f"DEBUG: gt_time_numeric unique values: {mdf['gt_time_numeric'].unique()}")
        print(f"DEBUG: Number of samples: {len(mdf)}")
        if split_col is None:
            row_data = add_res_to_df(results[dataset][method][column], dataset, method, None, column, experiment)
            rows_list.append(row_data)
        else:
            splits = results[dataset][method][column][split_col].unique()
            skip_corr = True if (experiment in ['out_dist']) else False
            for split in splits:

                sub_df = results[dataset][method][column][results[dataset][method][column][split_col] == split]
                row_data = add_res_to_df(sub_df, dataset, method, split, column, experiment, skip_corr=skip_corr)
                rows_list.append(row_data)

    new_df = pd.DataFrame(rows_list)
    new_df.set_index(['dataset', 'method', 'split', 'exp'], inplace=True)
        # Set the MultiIndex here

    return new_df, results


def plot_coarse_vs_gt_time(results, save_suffix="", methods = ['sceptic', 'ga_reg', 'nb_cumulative_logit'], exps = ['windowed_day_coarsened', 'windowed_day_overlapping'],
                           methods_short = ['sceptic', 'ga_reg', 'nb_cl'], exps_short = ['coarsened', 'overlapping']):
    
    
    os.makedirs('coarse_vs_gt_plots', exist_ok=True)

    if len(methods_short) != len(methods):
        raise ValueError("Length of methods_short must match length of methods")
    if len(exps_short) != len(exps):
        raise ValueError("Length of exps_short must match length of exps")


    unique_gt_times = results['mouse_embryo'][methods[0]][exps[0]]['gt_time_numeric'].unique()
    unique_coarse_windows = results['mouse_embryo'][methods[0]]['windowed_day_coarsened']['train_time_numeric'].unique()
    unique_overlapping_windows = results['mouse_embryo'][methods[0]]['windowed_day_overlapping']['train_time_numeric'].unique()

    # --- Aggregated plot combining all windows ---
    nrows = len(exps)
    ncols = len(methods)
    fig_agg, axs_agg = plt.subplots(nrows, ncols, figsize=(0.5*TEXT_WIDTH / 72.27,  (0.618*(0.66*0.8)*TEXT_WIDTH) / 72.27), layout="constrained")
    all_handles_agg = {}

    for i, j in itertools.product(range(nrows), range(ncols)):
        exp = exps[i]
        method = methods[j]
        ax = axs_agg[i, j]

        df = results['mouse_embryo'][method][exp]

       # Filter to windows that exist in both coarse and overlapping
        common_windows = sorted(set(unique_coarse_windows) & set(unique_overlapping_windows))
        df_common = df[df['train_time_numeric'].isin(common_windows)]

        lower_offsets = []
        upper_offsets = []

        for train_window in common_windows:
            df_tw = df_common[df_common['train_time_numeric'] == train_window].copy()
            df_tw = df_tw.dropna(subset=['gt_time_numeric', 'pred_time'])
            if df_tw.empty:
                print(f"No valid predictions for train window {train_window} in {method} {exp}")
                continue

            # GT windows observed for this train window (expected: 2)
            gt_vals = sorted(df_tw['gt_time_numeric'].unique())
            
            # Check gt_vals is correct size
            if len(gt_vals) != 2:
                if len(gt_vals) == 0:
                    print(f"No GT values for train window {train_window} in {method} {exp}")
                    print(gt_vals)
                    continue
                elif len(gt_vals) == 1:
                    print(f"Insufficient GT values (1 found expected 2) for train window {train_window} in {method} {exp}")
                    print(gt_vals)
                    continue
                elif len(gt_vals) > 2:
                    print("Too many gt vals. Expected 2")
                    print(gt_vals)
                    continue

            # Pick lower/upper GT relative to train window; fallback to min/max
            lower_candidates = [g for g in gt_vals if g <= train_window]
            upper_candidates = [g for g in gt_vals if g >= train_window]

            lower_gt = max(lower_candidates) if lower_candidates else min(gt_vals)
            upper_gt = min(upper_candidates) if upper_candidates else max(gt_vals)

            lower_df = df_tw[df_tw['gt_time_numeric'] == lower_gt]
            upper_df = df_tw[df_tw['gt_time_numeric'] == upper_gt]

            lower_offsets.extend((lower_df['pred_time'] - train_window).to_numpy())
            upper_offsets.extend((upper_df['pred_time'] - train_window).to_numpy())

        # Plot lower/upper density histograms
        if len(lower_offsets) > 0:
            lower_counts, lower_edges = np.histogram(
                np.asarray(lower_offsets), bins=100, range=(-5, 5), density=True
            )
            lower_line = ax.stairs(lower_counts, lower_edges, alpha=0.9, label='lower_gt')
            if 'lower_gt' not in all_handles_agg:
                all_handles_agg['lower_gt'] = lower_line

        if len(upper_offsets) > 0:
            upper_counts, upper_edges = np.histogram(
                np.asarray(upper_offsets), bins=100, range=(-5, 5), density=True
            )
            upper_line = ax.stairs(upper_counts, upper_edges, alpha=0.9, label='upper_gt')
            if 'upper_gt' not in all_handles_agg:
                all_handles_agg['upper_gt'] = upper_line

        ax.set_xlim(-3, 3)

    # Add row labels (exp names) on the left side
    for i, exp_short in enumerate(exps_short):
        axs_agg[i, 0].set_ylabel(exp_short, fontsize=FONT_SIZE_LABEL)

     # Add row labels (exp names) on the left side
    for j, method_short in enumerate(methods_short):
        axs_agg[0, j].set_title(method_short, fontsize=FONT_SIZE_TITLE)

    axs_agg[1, 1].set_xlabel('Pred - Train Window', fontsize=FONT_SIZE_LABEL)

    # Use one shared legend for the aggregated figure, anchored off the bottom-right.
    if all_handles_agg:
        legend_order = ['lower_gt', 'upper_gt']
        handles = [all_handles_agg[k] for k in legend_order if k in all_handles_agg]
        labels = [k for k in legend_order if k in all_handles_agg]
        box = axs_agg[1, 1].get_position()

        fig_agg.legend(
            handles,
            labels,
            loc='upper left',
            bbox_to_anchor=(0, box.y0),
            frameon=False,
            fontsize=FONT_SIZE_LEGEND,
            ncol=2
        )

    # Leave extra bottom margin so the external legend is visible.
    fig_agg.savefig(f"coarse_vs_gt_plots/coarse_vs_gt_aggregated_{save_suffix}.pdf", bbox_inches='tight')


    for o_window in unique_overlapping_windows:
        if o_window not in unique_coarse_windows:
            continue
        nrows = len(exps)
        ncols = len(methods)
        fig, axs = plt.subplots(nrows, ncols, figsize=(0.5*TEXT_WIDTH / 72.27,  (0.618*(0.66*0.8)*TEXT_WIDTH) / 72.27), layout="constrained")
        all_handles = {}
        for i, j in itertools.product(range(nrows), range(ncols)):
            exp = exps[i]
            method = methods[j]
            ax = axs[i, j]

            df = results['mouse_embryo'][method][exp]
            df_o = df[df['train_time_numeric'] == o_window]

            for time in unique_gt_times:
                df_gt = df_o[df_o['gt_time_numeric'] == time]
            
                pred_time = df_gt['pred_time'].dropna()
                if pred_time.empty:
                    continue
                bin_counts, bin_edges = np.histogram(pred_time, bins=100, density=True)  # 30 bins
        
                line = ax.stairs(bin_counts, bin_edges, alpha=1, label=time)
                ax.axvline(x=o_window, color='black', linestyle='--', label='train_center')
                ax.set_xlim(o_window - 5, o_window + 5)
                if time not in all_handles:
                    all_handles[time] = line

        # Add column titles (method names) to top row
        for j, method in enumerate(methods_short):
            axs[0, j].set_title(method)

        # Add row labels (exp names) on the left side
        for i, exp in enumerate(exps_short):
            axs[i, 0].set_ylabel(exp)

        if all_handles:
            handles = [all_handles[t] for t in sorted(all_handles.keys())]
            labels = [str(t) for t in sorted(all_handles.keys())]
            axs[0,0].legend(handles, labels, 
                    loc='center left',
                    ncol=1,
                    title='GT Time',
                    fontsize=FONT_SIZE_LEGEND)
        else:
            fig.tight_layout()

        fig.savefig(f"coarse_vs_gt_plots/coarse_vs_gt_{o_window}_{save_suffix}.pdf", bbox_inches='tight')
        plt.close(fig)


    #Plot last window
    print("\n\nPlotting last two windows together...\n\n")
    unique_gt_times = np.sort(np.array(unique_gt_times, dtype=float))
    last_two_windows = unique_gt_times[-2:]

    print(f"Last two GT windows: {last_two_windows}")

    nrows = len(exps)
    ncols = len(methods)
    fig_last, axs_last = plt.subplots(nrows, ncols, figsize=(0.5*TEXT_WIDTH / 72.27,  (0.618*(0.66*0.8)*TEXT_WIDTH) / 72.27), layout="constrained")
    all_handles = {}

    for i, j in itertools.product(range(nrows), range(ncols)):
        exp = exps[i]
        method = methods[j]
        ax = axs_last[i, j]

        df = results['mouse_embryo'][method][exp]

        for time in last_two_windows:
            df_gt = df[df['gt_time_numeric'] == time]
        
            pred_time = df_gt['pred_time'].dropna()
            if pred_time.empty:
                continue
            bin_counts, bin_edges = np.histogram(pred_time, bins=100, density=True)  # 30 bins
    
            line = ax.stairs(bin_counts, bin_edges, alpha=1, label=time)
            c = line.get_edgecolor()
            if np.ndim(c) > 1:
                c = c[0]  # some matplotlib versions return shape (1, 4)

            ax.axvline(time, linestyle='--', color=c)
            if time not in all_handles:
                all_handles[time] = line

        axs_last[i, j].set_xlim(14, 24)

    # Add column titles (method names) to top row
    for j, method in enumerate(methods_short):
        axs_last[0, j].set_title(method, fontsize=FONT_SIZE_TITLE)

    # Add row labels (exp names) on the left side
    for i, exp_short in enumerate(exps_short):
        axs_last[i, 0].set_ylabel(exp_short, fontsize=FONT_SIZE_LABEL)


    axs_last[1, 1].set_xlabel('predicted time')

    if all_handles:
        box = axs_last[1, 1].get_position()

        handles = [all_handles[t] for t in sorted(all_handles.keys())]
        labels = [str(t) for t in sorted(all_handles.keys())]
    
        fig_last.legend(
            handles,
            labels,
            loc='upper left',
            bbox_to_anchor=(0, box.y0),
            frameon=False, 
            fontsize=FONT_SIZE_LEGEND,
            ncol=2
        )

    fig_last.savefig(f"coarse_vs_gt_plots/final_windows_{save_suffix}.pdf", bbox_inches='tight')
    plt.close(fig_last)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment metrics")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--records", required=True, help="path to YAML file with records")
    parser.add_argument("--metrics", required=False, help="Comma-separated list of metrics to plot", default="mse,accuracy")
    parser.add_argument("--method_colours", required=False, help="JSON string of method color indices", default=None)
    parser.add_argument("--colormap", required=False, help="Matplotlib colormap name", default='tab20')
    parser.add_argument("--publishdir", required=False, help="Directory to save publication-ready plots", default=FileNotFoundError)
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

    text_width = 505

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

    anndata = sc.read_h5ad('/mnt/mr01-data01/j72687wm/ordVae_paper/nextflow_mouse/results/data/mouse_embryo_processed.h5ad')
    
    time_columns = {}
    col_base = list(set([rec['column'] for rec in records if 'column_suffix' in rec]))
    for cb in col_base:
        tmp_col = list(set([cb + rec.get('column_suffix', '') for rec in records]))
        tmp_col.sort()
        time_columns[cb] =  tmp_col
    print(f'time_columns: {time_columns}')
    mapping_df_per_col = {}
    WINDOW_MAP_PER_COL = {}
    for key in time_columns:
        mapping_df_per_col[key] = anndata.obs[time_columns[key]].drop_duplicates()
        WINDOW_MAP_PER_COL[key] = dict(zip(*[mapping_df_per_col[key][col] for col in mapping_df_per_col[key].columns]))
    
    UNIQUE_EDGES_PER_COL = {}
    UNIQUE_EDGES_STR_PER_COL = {}
    for col in time_columns:
        _, UNIQUE_EDGES_PER_COL[col], _ = string_to_splits_np(anndata.obs[col], inf_edge=True)
        UNIQUE_EDGES_STR_PER_COL[col] = ['_'.join(window) for window in UNIQUE_EDGES_PER_COL[col].astype(str)]


    print(f'\nwindow map per col {WINDOW_MAP_PER_COL}\n')

    print(f'\nunique edges per col {UNIQUE_EDGES_PER_COL}\n')
    print(f'\nunique edges str per col {UNIQUE_EDGES_STR_PER_COL}\n')

    #%% Analyze TRAIN data only

    print("\n" + "="*50)
    print("Analyzing TRAIN data only...")
    print("="*50)
    os.mkdir('train_plots')
    os.chdir('train_plots')

    missing_counter_train = {}
    missing_records_train = []

    pubdir = args.publishdir
    if pubdir is not None:
        Path(pubdir).mkdir(parents=True, exist_ok=True)

    if (pubdir is not None) and os.path.isfile(os.path.join(pubdir, 'results_train.pkl')):
        print(f"Loading existing results from {pubdir}/results_train.pkl")
        with open(os.path.join(pubdir, 'results_train.pkl'), 'rb') as f:
            df_train, results_train = pkl.load(f)
        
    else:
        df_train, results_train = add_records_to_df(
            records, missing_counter_train, missing_records_train, experiment,
            gt_window_series=anndata.obs['windowed_day_base'],
            split_col=None,
            filter_split='train'
        )
        if pubdir is not None:
            with open(os.path.join(pubdir, 'results_train.pkl'), 'wb') as f:
                pkl.dump((df_train, results_train), f)

    

    # When using desired exp order and exps short, desired order should correspond to the modified exp name
    desired_exp_order = ['base', 'coarsened', 'overlapping']

    exps_short = {
        'windowed_day_base': 'base',
        'windowed_day_coarsened': 'coarsened',
        'windowed_day_overlapping': 'overlapping'
    }

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

    if not df_train.empty:
        df_train['accuracy'] = df_train['accuracy_gt_windows']
        df_train.to_csv(f"combined_metrics_{experiment}_train.csv")
        
        methods_train = df_train.index.get_level_values('method').unique().tolist()
        metric_fig_size = (0.5*TEXT_WIDTH / 72.27, (0.618*0.5*TEXT_WIDTH) / 72.27)
        for metric in metrics:
            plot_boxplot_metric_by_sub_exp(df_train, experiment, methods_train, metric=metric,
                                           title=f"{metric}_by_col_{experiment}_train.pdf",
                                           desired_exp_order=desired_exp_order,
                                           exps_short=exps_short,
                                           method_names_short=method_names_short,
                                           figsize=metric_fig_size)

        rank_fig_size = (0.5*TEXT_WIDTH / 72.27, (0.75*0.618*0.5*TEXT_WIDTH) / 72.27)

        exps_order = ['base', 'coarsened', 'overlapping']

        plot_rank_by_column_root(df_train, experiment, metrics,
                                 method_color_indices=method_colours_indices,
                                 colormap=args.colormap,
                                 title=f"rank_by_exp_{experiment}_train.pdf",
                                 exps_short=exps_short,
                                 method_names_short=method_names_short,
                                 figsize=rank_fig_size,
                                 exps_order=exps_order)

        plot_coarse_vs_gt_time(results_train, save_suffix="train")

         # After processing all records, write missing reports
    if  missing_records_train:
        df_missing = pd.DataFrame(missing_records_train)
        df_missing.to_csv(f"missing_files_{experiment}_train.csv", index=False)

        summary = (
            df_missing.groupby(["dataset", "method", "exp"])
            .size()
            .reset_index(name="missing_count")
            .sort_values(["missing_count", "dataset", "method", "exp"], ascending=[False, True, True, True])
        )
        summary.to_csv(f"missing_summary_{experiment}_train.csv", index=False)
        print(f"Wrote {len(df_missing)} missing entries to missing_files_{experiment}_train.csv and summary to missing_summary_{experiment}_train.csv")
    else:
        print("No missing result files detected.")

    os.chdir('..')

    print("generating plots without splits for out_dist experiment...")
    missing_counter_ns = {}
    missing_records_ns = []  # add

    #%% Analyze TEST data only
    # plot the distribution of predicted pseudo-times for each method, colored by ground truth time class for all test data
    missing_counter_test = {}
    missing_records_test = []
    
    if (pubdir is not None) and os.path.isfile(os.path.join(pubdir, 'results_test.pkl')):
        print(f"Loading existing results from {pubdir}/results_test.pkl")
        with open(os.path.join(pubdir, 'results_test.pkl'), 'rb') as f:
            df_test, results_test = pkl.load(f)
        
    else:
        df_test, results_test = add_records_to_df(
            records, missing_counter_test, missing_records_test, experiment,
            gt_window_series=anndata.obs['windowed_day_base'],
            split_col=None,
            filter_split='test'
        )
        if pubdir is not None:
            with open(os.path.join(pubdir, 'results_test.pkl'), 'wb') as f:
                pkl.dump((df_test, results_test), f)

    for dataset in results_test:
        for method in results_test[dataset]:
            for column in results_test[dataset][method]:
                os.makedirs(f'psuedo_time_dist_plots/{dataset}/{method}/', exist_ok=True)
                plot_psuedo_time_dist(results_test[dataset][method][column], method , save_loc=f'psuedo_time_dist_plots/{dataset}/{method}/pseudo_time_dist_{column}.pdf')
                plt.close()

    del df_test, results_test

    ##%% Analyze test in dist and out_dist separately, since for out_dist we only have test data and want to generate separate plots for this
    #max(), 10, 13, 17, 17

    print("\n" + "="*50)
    print("Analyzing TEST data only...")
    print("="*50)

    os.mkdir('test_indist_plot')
    os.chdir('test_indist_plot')

    held_out = [10.0, 13.0, 17.0, anndata.obs['day_num'].max()]
    held_out_interpolation = anndata.obs['day_num'].isin([10.0, 13.0, 17.0,])
    held_out_extrapolation = anndata.obs['day_num'].isin([anndata.obs['day_num'].max()])
    held_out = anndata.obs['day_num'].isin(held_out)

    in_dist_idx = anndata.obs[~held_out].index
    extrapolation_idx = anndata.obs[held_out_extrapolation].index
    interpolation_idx = anndata.obs[held_out_interpolation].index

    # In dist

    missing_counter_test = {}
    missing_records_test = []
    if (pubdir is not None) and os.path.isfile(os.path.join(pubdir, 'results_test_in_dist.pkl')):
        print(f"Loading existing results from {pubdir}/results_test_in_dist.pkl")
        with open(os.path.join(pubdir, 'results_test_in_dist.pkl'), 'rb') as f:
            df_test_in_dist, results_test_in_dist = pkl.load(f)
        
    else:
        df_test_in_dist, results_test_in_dist = add_records_to_df(
                records, missing_counter_test, missing_records_test, experiment,
                gt_window_series=anndata.obs['windowed_day_base'],
                split_col=None,
                filter_split='test',
                idx_filter=in_dist_idx
            )
        if pubdir is not None:
            with open(os.path.join(pubdir, 'results_test_in_dist.pkl'), 'wb') as f:
                pkl.dump((df_test_in_dist, results_test_in_dist), f)
    
    if not df_test_in_dist.empty:
        df_test_in_dist['accuracy'] = df_test_in_dist['accuracy_gt_windows']
        df_test_in_dist.to_csv(f"combined_metrics_{experiment}_test_in_dist.csv")
        
        methods_test = df_test_in_dist.index.get_level_values('method').unique().tolist()
        for metric in metrics:
            plot_boxplot_metric_by_sub_exp(df_test_in_dist, experiment, methods_test, metric=metric,
                                           title=f"{metric}_by_col_{experiment}_test_in_dist.pdf",
                                           desired_exp_order=desired_exp_order,
                                            exps_short=exps_short,
                                           method_names_short=method_names_short,
                                           figsize=metric_fig_size)
            
        plot_rank_by_column_root(df_test_in_dist, experiment, metrics,
                                 method_color_indices=method_colours_indices,
                                 colormap=args.colormap,
                                 title=f"rank_by_exp_{experiment}_test_in_dist.pdf",
                                 exps_short=exps_short,
                                 method_names_short=method_names_short,
                                 figsize=rank_fig_size,
                                 exps_order=exps_order)
        
        plot_coarse_vs_gt_time(results_test_in_dist, save_suffix="test_in_dist")

    
    # After processing all records, write missing reports
    if  missing_records_test:
        df_missing = pd.DataFrame(missing_records_test)
        df_missing.to_csv(f"missing_files_{experiment}_test_in_dist.csv", index=False)

        summary = (
            df_missing.groupby(["dataset", "method", "exp"])
            .size()
            .reset_index(name="missing_count")
            .sort_values(["missing_count", "dataset", "method", "exp"], ascending=[False, True, True, True])
        )
        summary.to_csv(f"missing_summary_{experiment}_test_in_dist.csv", index=False)
        print(f"Wrote {len(df_missing)} missing entries to missing_files_{experiment}_test_in_dist.csv and summary to missing_summary_{experiment}_test_in_dist.csv")
    else:
        print("No missing result files detected.")

    os.chdir('..')

    # extrap

    os.mkdir('test_extrapolation_plot')
    os.chdir('test_extrapolation_plot')

    missing_counter_test = {}
    missing_records_test = []

    if (pubdir is not None) and os.path.isfile(os.path.join(pubdir, 'results_test_out_dist_extrap.pkl')):
        print(f"Loading existing results from {pubdir}/results_test_out_dist_extrap.pkl")
        with open(os.path.join(pubdir, 'results_test_out_dist_extrap.pkl'), 'rb') as f:
            df_test_out_dist_extrap, results_test_out_dist_extrap = pkl.load(f)
        
    else:
        df_test_out_dist_extrap, results_test_out_dist_extrap = add_records_to_df(
            records, missing_counter_test, missing_records_test, experiment,
            gt_window_series=anndata.obs['windowed_day_base'],
            split_col=None,
            filter_split='test',
            idx_filter=extrapolation_idx
        )
        if pubdir is not None:
            with open(os.path.join(pubdir, 'results_test_out_dist_extrap.pkl'), 'wb') as f:
                pkl.dump((df_test_out_dist_extrap, results_test_out_dist_extrap), f)



    if not df_test_out_dist_extrap.empty:
        df_test_out_dist_extrap['accuracy'] = df_test_out_dist_extrap['accuracy_gt_windows']
        df_test_out_dist_extrap.to_csv(f"combined_metrics_{experiment}_test_out.csv")
        
        methods_test = df_test_out_dist_extrap.index.get_level_values('method').unique().tolist()
        metrics = ['mse', 'accuracy']
        for metric in metrics:
            plot_boxplot_metric_by_sub_exp(df_test_out_dist_extrap, experiment, methods_test, metric=metric,
                                           title=f"{metric}_by_col_{experiment}_test_extrap.pdf",
                                           desired_exp_order=desired_exp_order,
                                           exps_short=exps_short,
                                           method_names_short=method_names_short,
                                           figsize=metric_fig_size)
            
        plot_rank_by_column_root(df_test_out_dist_extrap, experiment, metrics,
                                 method_color_indices=method_colours_indices,
                                 colormap=args.colormap,
                                 title=f"rank_by_exp_{experiment}_test_extrap.pdf",
                                 exps_short=exps_short,
                                 method_names_short=method_names_short,
                                 figsize=rank_fig_size,
                                 exps_order=exps_order)

        plot_coarse_vs_gt_time(results_test_out_dist_extrap, save_suffix="test_extrap")

        
     # After processing all records, write missing reports
    if  missing_records_test:
        df_missing = pd.DataFrame(missing_records_test)
        df_missing.to_csv(f"missing_files_{experiment}_test_extrap.csv", index=False)

        summary = (
            df_missing.groupby(["dataset", "method", "exp"])
            .size()
            .reset_index(name="missing_count")
            .sort_values(["missing_count", "dataset", "method", "exp"], ascending=[False, True, True, True])
        )
        summary.to_csv(f"missing_summary_{experiment}_test_extrap.csv", index=False)
        print(f"Wrote {len(df_missing)} missing entries to missing_files_{experiment}_test_extrap.csv and summary to missing_summary_{experiment}_test_extrap.csv")
    else:
        print("No missing result files detected.")

    os.chdir('..')
    # interpolation

    os.mkdir('test_interpolation_plot')
    os.chdir('test_interpolation_plot')

    missing_counter_test = {}
    missing_records_test = []

    if (pubdir is not None) and os.path.isfile(os.path.join(pubdir, 'results_test_out_dist_interp.pkl')):
        print(f"Loading existing results from {pubdir}/results_test_out_dist_interp.pkl")
        with open(os.path.join(pubdir, 'results_test_out_dist_interp.pkl'), 'rb') as f:
            df_test_out_dist_interp, results_test_out_dist_interp = pkl.load(f)
        
    else:
        df_test_out_dist_interp, results_test_out_dist_interp = add_records_to_df(
            records, missing_counter_test, missing_records_test, experiment,
            gt_window_series=anndata.obs['windowed_day_base'],
            split_col=None,
            filter_split='test',
            idx_filter=interpolation_idx
        )
        if pubdir is not None:
            with open(os.path.join(pubdir, 'results_test_out_dist_interp.pkl'), 'wb') as f:
                pkl.dump((df_test_out_dist_interp, results_test_out_dist_interp), f)



    if not df_test_out_dist_interp.empty:
        df_test_out_dist_interp['accuracy'] = df_test_out_dist_interp['accuracy_gt_windows']
        df_test_out_dist_interp.to_csv(f"combined_metrics_{experiment}_test_interpol.csv")
        
        methods_test = df_test_out_dist_interp.index.get_level_values('method').unique().tolist()
        metrics = ['mse', 'accuracy']
        for metric in metrics:
            plot_boxplot_metric_by_sub_exp(df_test_out_dist_interp, experiment, methods_test, metric=metric,
                                           title=f"{metric}_by_col_{experiment}_test_interpol.pdf",
                                           desired_exp_order=desired_exp_order,
                                           exps_short=exps_short,
                                           method_names_short=method_names_short,
                                           figsize=metric_fig_size)
        
        plot_rank_by_column_root(df_test_out_dist_interp, experiment, metrics,
                                 method_color_indices=method_colours_indices,
                                 colormap=args.colormap,
                                 title=f"rank_by_exp_{experiment}_test_interpol.pdf",
                                 exps_short=exps_short,
                                 method_names_short=method_names_short,
                                 figsize=rank_fig_size,
                                 exps_order=exps_order)

        plot_coarse_vs_gt_time(results_test_out_dist_interp, save_suffix="test_interp")
        plot_coarse_vs_gt_time(results_test_out_dist_interp, save_suffix="test_interp_psupertime", methods=['pypsupertime', 'sceptic', 'ga_reg', 'nb_ord'],
                               exps=['windowed_day_base', 'windowed_day_coarsened', 'windowed_day_overlapping'],
                               methods_short=['psupertime', 'sceptic', 'ga_reg', 'nb_ord'], exps_short=['base', 'coarsened', 'overlapping'])
        
     # After processing all records, write missing reports
    if  missing_records_test:
        df_missing = pd.DataFrame(missing_records_test)
        df_missing.to_csv(f"missing_files_{experiment}_test_interpol.csv", index=False)

        summary = (
            df_missing.groupby(["dataset", "method", "exp"])
            .size()
            .reset_index(name="missing_count")
            .sort_values(["missing_count", "dataset", "method", "exp"], ascending=[False, True, True, True])
        )
        summary.to_csv(f"missing_summary_{experiment}_test_interpol.csv", index=False)
        print(f"Wrote {len(df_missing)} missing entries to missing_files_{experiment}_test_interpol.csv and summary to missing_summary_{experiment}_test_interpol.csv")
    else:
        print("No missing result files detected.")