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
                    'accuracy_train_windows': calculate_overlapping_accuracy(sub_df['gt_time_numeric'], sub_df['pred_class_overlapping_train_windows']),
                    'accuracy_gt_windows': accuracy_score(sub_df['gt_time_numeric'].astype(str), sub_df['pred_class_overlapping_gt_windows'].astype(str))})
        
        return row


def add_records_to_df(records, missing_counter, missing_records, experiment, split_col='split', gt_window_series=None):
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
        mdf['train_window'] = mdf['gt_window']
       
        if gt_window_series is not None:
            mdf['gt_window'] = gt_window_series[mdf.index]

        if (mdf['split'].dtype not in [int, float]) and (mdf['split'].iloc[0] != 'all_data'):
            mdf['split'] = mdf['split'].map(WINDOW_MAP_PER_COL[column])

        # Ensure each cell has a categorical and a numeric representation of the ground truth time
        if (mdf['gt_window'].dtype == object) | (mdf['gt_window'].dtype.name == 'category'):
            mdf['gt_time_numeric'] = mdf['gt_window'].map(WINDOW_MAP_PER_COL['windowed_day_exp1'])
            mdf['gt_time_numeric'] = pd.to_numeric(mdf['gt_time_numeric'])
            if mdf['gt_time_numeric'].isna().any():
                print(f"Warning: NaN values found in gt_time_numeric for {dataset}/{method}/{column}. Check WINDOW_MAP_PER_COL and gt_window values.")
                print(WINDOW_MAP_PER_COL['windowed_day_exp1'])
                print(mdf['gt_window'])
                print(mdf['gt_time_numeric'])
                raise ValueError("NaN values found in gt_time_numeric after mapping. Check the warning messages for details.")
            
        elif mdf['gt_window'].dtype in [float, int]:
            mdf['gt_time_numeric'] = mdf['gt_window']
        else:
            print(f"Unknown dtype for gt_window: {mdf['gt_window'].dtype}")
            mdf['gt_time_numeric'] = mdf['gt_window']
        
        # Create inverse mapping for gt_window to get categorical time class back from numeric
        inv_map = {v: k for k, v in WINDOW_MAP_PER_COL['windowed_day_exp1'].items()}

        if mdf['gt_window'].dtype == object:
            mdf['gt_time_class'] = mdf['gt_window']
        elif mdf['gt_window'].dtype in [float, int]:
            mdf['gt_time_class'] = mdf['gt_window'].map(inv_map)
        else:
            print(f"Unknown dtype for gt_window: {mdf['gt_window'].dtype}")
            mdf['gt_time_class'] = mdf['gt_window']

        # Ensure each cell has a categorical and a numeric representation of label used during training
        if mdf['train_window'].dtype == object:
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
        mdf['pred_class_overlapping_gt_windows'] = find_windows_unique(UNIQUE_EDGES_PER_COL['windowed_day_exp1'],
                                                                                                    UNIQUE_EDGES_STR_PER_COL['windowed_day_exp1'],
                                                                                                    mdf['pred_time'])
        mdf['pred_class_overlapping_gt_windows']  = mdf['pred_class_overlapping_gt_windows'].map(WINDOW_MAP_PER_COL['windowed_day_exp1'])
        
        if dataset not in results:
            results[dataset] = {}
        if method not in results[dataset]:
            results[dataset][method] = {}

        results[dataset][method][column] = mdf

        
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

def plot_coarse_vs_gt_time(results):
    """
    Generates density plots of predicted time minus train window for predictions grouped by GT time, for each experiment and method. Also generates an aggregated plot combining all windows.
    """

    os.mkdir('coarse_vs_gt_plots')
    methods = ['sceptic', 'temporalVAE', 'nb_cumulative_logit']
    methods_short = ['sceptic', 'tempVAE', 'nb_cl']
    exps = ['windowed_day_exp2', 'windowed_day_exp3']
    exps_short = ['coarsened', 'overlapping']

    unique_gt_times = results['wadot'][methods[0]][exps[0]]['gt_time_numeric'].unique()
    unique_coarse_windows = results['wadot'][methods[0]]['windowed_day_exp2']['train_time_numeric'].unique()
    unique_overlapping_windows = results['wadot'][methods[0]]['windowed_day_exp3']['train_time_numeric'].unique()

    # --- Aggregated plot combining all windows ---
    nrows = len(exps)
    ncols = len(methods)
    fig_agg, axs_agg = plt.subplots(nrows, ncols, figsize=(0.5*TEXT_WIDTH / 72.27,  (0.618*(0.66*0.8)*TEXT_WIDTH) / 72.27), layout="constrained")
    all_handles_agg = {}

    for i, j in itertools.product(range(nrows), range(ncols)):
        exp = exps[i]
        method = methods[j]
        ax = axs_agg[i, j]

        df = results['wadot'][method][exp]

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

        ax.set_xlim(-5, 5)

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
    fig_agg.savefig(f"coarse_vs_gt_plots/coarse_vs_gt_aggregated.pdf", bbox_inches='tight')


    for o_window in unique_overlapping_windows:
        if o_window not in unique_coarse_windows:
            continue
        nrows = len(exps)
        ncols = len(methods)
        fig, axs = plt.subplots(nrows, ncols, figsize=(0.5*TEXT_WIDTH / 72.27, (0.618*(0.66*0.8)*TEXT_WIDTH) / 72.27), layout="constrained")
        all_handles = {}
        for i, j in itertools.product(range(nrows), range(ncols)):
            exp = exps[i]
            method = methods[j]
            ax = axs[i, j]

            df = results['wadot'][method][exp]
            df_o = df[df['train_time_numeric'] == o_window]

            for time in unique_gt_times:
                df_gt = df_o[df_o['gt_time_numeric'] == time]
            
                pred_time = df_gt['pred_time'].dropna()
                if pred_time.empty:
                    continue
                bin_counts, bin_edges = np.histogram(pred_time, bins=100, density=True)  # 30 bins
        
                line = ax.stairs(bin_counts, bin_edges, alpha=1, label=time)
                ax.set_xlim(o_window - 5, o_window + 5)
                if time not in all_handles:
                    all_handles[time] = line

        # Add column titles (method names) to top row
        for j, method in enumerate(methods_short):
            axs[0, j].set_title(method, fontsize=FONT_SIZE_TITLE)

        # Add row labels (exp names) on the left side
        for i, exp_short in enumerate(exps_short):
            axs[i, 0].set_ylabel(exp_short, fontsize=FONT_SIZE_LABEL)

        axs[1, 1].set_xlabel('predicted time')

        if all_handles:
            box = axs[1, 1].get_position()

            handles = [all_handles[t] for t in sorted(all_handles.keys())]
            labels = [str(t) for t in sorted(all_handles.keys())]
        
            fig.legend(
                handles,
                labels,
                loc='upper left',
                bbox_to_anchor=(0, box.y0),
                frameon=False,
                fontsize=FONT_SIZE_LEGEND,
                ncol=2
            )

        fig.savefig(f"coarse_vs_gt_plots/coarse_vs_gt_{o_window}.pdf", bbox_inches='tight')
        plt.close(fig)


    # Plot last window
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

        df = results['wadot'][method][exp]

        for time in last_two_windows:
            df_gt = df[df['gt_time_numeric'] == time]
        
            pred_time = df_gt['pred_time'].dropna()
            if pred_time.empty:
                continue
            bin_counts, bin_edges = np.histogram(pred_time, bins=100, density=True)  # 30 bins
    
            line = ax.stairs(bin_counts, bin_edges, alpha=1, label=time)
            # Reuse histogram edge color for the GT line
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

    fig_last.savefig(f"coarse_vs_gt_plots/final_windows.pdf", bbox_inches='tight')
    plt.close(fig_last)


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

    anndata = sc.read_h5ad('/mnt/mr01-data01/j72687wm/ordVae_paper/nextflow_wadot/results/data/wadot_processed.h5ad')
    
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

    pubdir = args.publishdir
    if (pubdir is not None) and os.path.isfile(os.path.join(pubdir, 'results.pkl')):
        print(f"Loading existing results from {pubdir}/results.pkl")
        with open(os.path.join(pubdir, 'results.pkl'), 'rb') as f:
            df, results = pkl.load(f)
        
    else:
        df, results = add_records_to_df(records, missing_counter, missing_records, experiment, gt_window_series=anndata.obs['windowed_day_exp1'])
        if pubdir is not None:
            with open(os.path.join(pubdir, 'results.pkl'), 'wb') as f:
                pkl.dump((df, results), f)

    df['accuracy'] = df['accuracy_gt_windows']

    output_path = Path(f"combined_metrics_{experiment}.csv")
    df.to_csv(output_path)
    print(f"\nAnalysis results saved to {output_path}\n")

    # Generate plots
    #plot spearman by exp

    # use named level for clarity
    if 'method' in df.index.names:
        methods = df.index.get_level_values('method').unique().tolist()
    else:
        methods = df.index.get_level_values(1).unique().tolist()

    exps_short = {
        'windowed_day_exp1': 'base',
        'windowed_day_exp2': 'coarsened',
        'windowed_day_exp3': 'overlapping'
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
    
    metric_fig_size = (0.5*TEXT_WIDTH / 72.27, (0.618*0.5*TEXT_WIDTH) / 72.27)
    for metric in metrics:
        plot_boxplot_metric_by_sub_exp(df, experiment, methods, metric=metric, title=f"{metric}_by_col_{experiment}.pdf",
                                       exps_short=exps_short, method_names_short=method_names_short, figsize=metric_fig_size)

    rank_fig_size = (0.5*TEXT_WIDTH / 72.27, (0.75*0.618*0.5*TEXT_WIDTH) / 72.27)  
    plot_rank_by_column_root(df, experiment, metrics, 
                        method_color_indices=method_colours_indices,
                        colormap=args.colormap,title=f"rank_by_exp_{experiment}.pdf",
                        exps_short=exps_short, method_names_short=method_names_short, figsize=rank_fig_size)

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
    
    # Generating psuedotime density plots
    for dataset in results:
        for method in results[dataset]:
            for column in results[dataset][method]:
                os.makedirs(f'psuedo_time_dist_plots/{dataset}/{method}/', exist_ok=True)
                plot_psuedo_time_dist(results[dataset][method][column], method , save_loc=f'psuedo_time_dist_plots/{dataset}/{method}/pseudo_time_dist_{column}.pdf')
                plt.close()

    # Generating coarse vs GT density plots
    plot_coarse_vs_gt_time(results)
    print("generating plots without splits for out_dist experiment...")
    missing_counter_ns = {}
    missing_records_ns = []  # add

    if experiment in ['out_dist']:
        if (pubdir is not None) and os.path.isfile(os.path.join(pubdir, 'results_no_split.pkl')):
            print(f"Loading existing results from {pubdir}/results_no_split.pkl")
            with open(os.path.join(pubdir, 'results_no_split.pkl'), 'rb') as f:
                df_no_split, results_no_split = pkl.load(f)
            
        else:
            df_no_split, results_no_split = add_records_to_df(records, missing_counter_ns, missing_records_ns, experiment, split_col=None,
                                                              gt_window_series=anndata.obs['windowed_day_exp1'])
            if pubdir is not None:
                with open(os.path.join(pubdir, 'results_no_split.pkl'), 'wb') as f:
                    pkl.dump((df_no_split, results_no_split), f)

        
        df_no_split['accuracy'] = df_no_split['accuracy_gt_windows']
        #print(df_no_split.head())

        # use named level for clarity
        if 'method' in df.index.names:
            methods = df.index.get_level_values('method').unique().tolist()
        else:
            methods = df.index.get_level_values(1).unique().tolist()
        metrics = ['spearman', 'kendall', 'pearson', 'mse', 'accuracy']
        for metric in metrics:
            plot_boxplot_metric_by_sub_exp(df_no_split, experiment, methods, metric=metric, title=f"{metric}_by_col_{experiment}_no_splits.pdf",
                                           exps_short=exps_short, method_names_short=method_names_short, figsize=metric_fig_size)
        
        plot_rank_by_column_root(df_no_split, experiment, metrics,
                                 method_color_indices=method_colours_indices,
                                 colormap=args.colormap,
                                 title=f"rank_by_exp_{experiment}_no_splits.pdf", figsize=rank_fig_size,
                                 exps_short=exps_short, method_names_short=method_names_short)

        print("generating plots based on extrapolation and interpolation...")
        print(df)
        split_values = df.index.get_level_values('split')
        # Ensure we treat them as floats for comparison
        split_floats = split_values.astype(float)
        

        # 2. Create Boolean masks using the MultiIndex values
        idx_vals = df.index.get_level_values('split').astype(float)
        exp_vals = df.index.get_level_values('exp')
        
        is_extrap = np.zeros(len(df), dtype=bool)  # Creates array of 740 False values

        for exp in exp_vals.unique():
            print(f"\nProcessing exp: {exp}")
            exp_mask = exp_vals == exp
            
            min_split_exp = split_floats[exp_mask].min()
            max_split_exp = split_floats[exp_mask].max()

            print(f"Timepoints detected - Min: {min_split_exp}, Max: {max_split_exp} for exp: {exp}")

            print(len(df), len(exp_mask), len(idx_vals))
            
            is_extrap[(((idx_vals == min_split_exp) | (idx_vals == max_split_exp)) & exp_mask)] = True
        is_interp = ~is_extrap

        # 3. Subset DataFrame
        df_extrap = df[is_extrap]
        df_interp = df[is_interp]

        print(df, df_extrap, df_interp)
        metrics = ['mse', 'accuracy']

        # 4. Plot Extrapolation
        if not df_extrap.empty:
            print(f"Plotting Extrapolation ({len(df_extrap)} rows)")
            for metric in metrics:
                plot_boxplot_metric_by_sub_exp(df_extrap, experiment, methods, metric=metric, 
                                            title=f"{metric}_by_col_{experiment}_extrapolation.pdf",
                                            exps_short=exps_short, method_names_short=method_names_short, figsize=metric_fig_size)
            plot_rank_by_column_root(df_extrap, experiment, metrics,
                                   method_color_indices=method_colours_indices,
                                   colormap=args.colormap,
                                   title=f"rank_by_exp_{experiment}_extrapolation.pdf", figsize=rank_fig_size,
                                   exps_short=exps_short, method_names_short=method_names_short)
        metrics = ['mse', 'accuracy']

        # 5. Plot Interpolation
        if not df_interp.empty:
            print(f"Plotting Interpolation ({len(df_interp)} rows)")
            for metric in metrics:
                plot_boxplot_metric_by_sub_exp(df_interp, experiment, methods, metric=metric, 
                                            title=f"{metric}_by_col_{experiment}_interpolation.pdf",
                                            exps_short=exps_short, method_names_short=method_names_short, figsize=metric_fig_size)
            plot_rank_by_column_root(df_interp, experiment, metrics, 
                                     method_color_indices=method_colours_indices,
                                    colormap=args.colormap,
                                    title=f"rank_by_exp_{experiment}_interpolation.pdf", figsize=rank_fig_size,
                                    exps_short=exps_short, method_names_short=method_names_short)

