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

def corr(x1, x2):
    sp_correlation, sp_p_value = spearmanr(x1, x2)
    ke_correlation, ke_p_value = kendalltau(x1, x2)
    p_correlation, p_p_value = pearsonr(x1, x2)

    if sp_correlation is None or np.isnan(sp_correlation):
        print(x1[:10], x2[:10])

        raise ValueError("Spearman correlation resulted in NaN.")

    return sp_correlation, ke_correlation, p_correlation


def rescale_to_range(y_pred, y_true, ood=False):
    if ood:
        # rescale only using train range
        y_scaled = np.full_like(y_pred, np.nan)
        y_pred_01, y_pred_99 = np.quantile(y_pred, 0.025), np.quantile(y_pred, 0.975)
        for timepoint in y_true.unique():
            test_mask = y_true == timepoint
            y_train_timepoint = y_true[~test_mask]
            y_train_min, y_train_max = np.min(y_train_timepoint), np.max(y_train_timepoint)
            
            y_scaled[test_mask] = temp_vae_rescale_fold(y_pred[test_mask], y_train_min, y_train_max, in_min=y_pred_01, in_max=y_pred_99)
        if np.isnan(y_scaled).any():
            print('Warning: Some predictions could not be rescaled due to missing timepoints in y_true. These will be set to NaN.')
        return y_scaled
    else:
        y_train_min, y_train_max = np.min(y_true), np.max(y_true)
        y_pred_01, y_pred_99 = np.quantile(y_pred, 0.01), np.quantile(y_pred, 0.99)
        return temp_vae_rescale_fold(y_pred, y_train_min, y_train_max, in_min=y_pred_01, in_max=y_pred_99)

def temp_vae_rescale_fold(y_pred, out_min, out_max, in_min=-5, in_max=5):
    y_pred_scaled = (y_pred - in_min) / (in_max - in_min)  # to [0,1]
    y_pred_scaled = y_pred_scaled * (out_max - out_min) + out_min
    return y_pred_scaled


def temp_vae_rescale_to_range(y_pred, y_true, ood=False):
    if ood:
        # rescale only using train range
        y_scaled = np.full_like(y_pred, np.nan)
        for timepoint in y_true.unique():
            test_mask = y_true == timepoint
            y_train_timepoint = y_true[~test_mask]
            y_train_min, y_train_max = np.min(y_train_timepoint), np.max(y_train_timepoint)
            y_scaled[test_mask] = temp_vae_rescale_fold(y_pred[test_mask], y_train_min, y_train_max)
        if np.isnan(y_scaled).any():
            print('Warning: Some predictions could not be rescaled due to missing timepoints in y_true. These will be set to NaN.')
        return y_scaled
    else:
        y_train_min = np.min(y_true)
        y_train_max = np.max(y_true)
        return temp_vae_rescale_fold(y_pred, y_train_min, y_train_max)



def plot_rank_by_column_root(df, experiment, metrics,  method_color_indices=None, colormap='tab20', title=f"rank_by_exp_experiment.pdf"):
    exps = df.index.get_level_values('exp').unique().tolist()
    if len(exps) == 0:
        print("No 'exp' levels found for rank plots.")
        raise Exception('no exp level values')
    
    from matplotlib import colormaps as cm
    cmap = cm.get_cmap(colormap)
    
    # Get all unique methods
    all_methods = df.index.get_level_values('method').unique().tolist()
    
    # Create color mapping
    if method_color_indices:
        # Use provided indices from config
        method_colors = {}
        for method in all_methods:
            if method in method_color_indices:
                method_colors[method] = cmap(method_color_indices[method])
            else:
                # Assign a default color for methods not in config
                print(f"Warning: Method '{method}' not in color config, using default")
                method_colors[method] = cmap(1)
    else:
        # Fallback: auto-assign based on sorted order
        method_colors = {method: cmap(i / len(all_methods)) 
                        for i, method in enumerate(sorted(all_methods))}

    #print(method_colors)

    n_exps = len(exps)
    if n_exps == 1:
        ncols=1
    elif n_exps == 2:
        ncols=2
    else:
        ncols = 3
    nrows = int(np.ceil(n_exps / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=((0.33*417/72.27) * ncols, (0.5*(417 * 0.618)/72.27) * nrows), squeeze=False)
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for i, exp in enumerate(exps):
        ax = axes[i]
        # select rows for this exp
        try:
            df_exp = df.xs(exp, level='exp')
        except KeyError:
            # fallback if level name isn't found

            raise KeyError(f"Level 'exp' not found in DataFrame index. Add dummy 'exp' level to DataFrame index if no sub exps.")
        
        df_tmp = df_exp.copy()
        cols = [c for c in metrics if c in df_tmp.columns]
        if (df_exp.empty) or (len(cols) == 0):
            ax.axis('off')
            continue

        if 'mse' in cols:
            # for mse lower is better, so invert
            df_tmp['mse'] = -df_tmp['mse']
        
        #print(f"tmp df: \n{df_tmp}")
        ranks_per_metric = {}
        for m in cols:
            if m not in df_tmp.columns:
                print(f"Warning: Metric '{m}' not found in DataFrame columns for exp '{exp}'. Skipping this metric.")
                continue
            t = df_tmp.reset_index()[['dataset', 'split', 'method', m]]
            t = t.dropna(subset=[m])
            if t['split'].isna().all():
                print(f"Warning: All 'split' values are NaN for exp '{exp}'. Ranking will be done only by dataset and method.")
                t['rank'] = t.groupby(['dataset'])[m].rank(ascending=False, method='average')
                
            elif t['dataset'].isna().all():
                print(f"Warning: All 'dataset' values are NaN for exp '{exp}'. Ranking will be done only by split and method.")
                t['rank'] = t.groupby(['split'])[m].rank(ascending=False, method='average')

            elif t['dataset'].isna().any() or t['split'].isna().any():
                print(f"Warning: Some 'dataset' or 'split' values are NaN for exp '{exp}'. These rows will be excluded from ranking.")
                t = t.dropna(subset=['dataset', 'split'])
                t['rank'] = t.groupby(['dataset', 'split'])[m].rank(ascending=False, method='average')
                
            else:
                # rank within each dataset×split
                t['rank'] = t.groupby(['dataset', 'split'])[m].rank(ascending=False, method='average')
                # average rank over units for each method
            ranks_per_metric[m] = t.groupby('method')['rank'].mean()

        ranks_df = pd.DataFrame(ranks_per_metric)  # methods × metrics
        ranks_df['ave'] = ranks_df.mean(axis=1)

        df_rank = ranks_df.sort_values('ave')

        #print(f"rank df: \n{df_rank}")

        # Plot each method with its assigned color
        for method in df_rank.index:
            if method in method_colors:
                color = method_colors[method]
            else:
                color = None  # fallback to default
            ax.plot(df_rank.T.index, df_rank.T[method], 
                    marker='o', label=method, color=color)
            
        if (exp is not None) & (not pd.isna(exp)):
            ax.set_title(str(exp))
        
        ax.set_xlabel("Metric")
        ax.set_ylabel("Rank")
        ax.grid(True)

        # set y ticks and invert so rank 1 is at top
        
        cols.append('ave')
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha='right')


    # hide any unused subplots
    if n_exps < len(axes):
        for j in range(n_exps, len(axes)):
            axes[j].axis('off')

    # add a single legend for the figure using handles from first subplot
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    index = title.rfind('.')
    fig.savefig(title[:index] + '_no_legend' + title[index:], bbox_inches='tight')

    if legend_handles:
        n_items = len(legend_labels)
        ncols = min(6, math.ceil(n_items / 5))
        fig.legend(
            legend_handles,
            legend_labels,
            bbox_to_anchor=(0.0, 0.0),  # bottom center
            loc='upper left',
            ncols=ncols,
            frameon=False,
            fontsize=6,
            columnspacing=0.5,
            handlelength=2.3,
            handletextpad=0.2,
            borderaxespad=0.0,
            labelspacing=0.4
        )
        # leave bottom margin for legend (adjust rect: left, bottom, right, top)
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    fig.savefig(title, bbox_inches='tight')


def plot_boxplot_metric_by_sub_exp(df, experiment, methods, metric='spearman', title=f"metric_by_col_experiment.pdf"):
    #print(df.head())

    if 'exp' in df.index.names:
        exp_levels = df.index.get_level_values('exp').unique()
    else:
        exp_levels = pd.Index([])
    
    if exp_levels.size == 1 or exp_levels.size == 0:
        print("No 'exp' levels found for boxplot. Single figure will be generated.")
        df = df.copy()
        df['data_type'] = df.index.get_level_values(0).str.split('_').str[1]
        df = df.set_index('data_type', append=True)

        fig, ax = plt.subplots(figsize=(417 / 72.27, 417 / 72.27 * 0.618))
        sns.boxplot(
            data=df.reset_index(),
            x="method",
            y=metric,
            flierprops={"marker": "x"},
            boxprops={"facecolor": (.3, .5, .7, .5)},
            medianprops={"color": "r", "linewidth": 1.5},
            order=methods,
            ax=ax,
        )

        ax.set_title(f"{metric} corr. by exp", fontdict={"fontweight": "bold"}, fontsize=10)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        plt.tight_layout()
        fig.savefig(title)
        return

    df = df.copy()
    df['data_type'] = df.index.get_level_values(0).str.split('_').str[1]
    df = df.set_index('data_type', append=True)
    df_long = df.reset_index()

    if metric not in df_long.columns:
        print(f"Metric '{metric}' not found in DataFrame columns.")
        return

    g = sns.FacetGrid(df_long, col='exp', sharey=True, col_wrap=3)
    g.map(sns.boxplot, "method", metric,
          flierprops={"marker": "x"},
          boxprops={"facecolor": (.3, .5, .7, .5)},
          medianprops={"color": "r", "linewidth": 1.5},
          order=methods)
    
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')

    # number of rows and cols
    if len(g.axes.shape) > 1:
        cols = g.axes.shape[1]  
        rows = g.axes.shape[0]          
    else:
        cols = g.axes.shape[0]
        rows = 1
    
    g.figure.subplots_adjust(top=0.85) # adjust the Figure in rp
    g.figure.suptitle(title, fontdict={"fontweight": "bold"}, fontsize=10)
    g.figure.set_size_inches((417/72.27),((rows * 417 )/72.27))
    g.set(xlabel=None, ylabel=None)
    plt.tight_layout()
    g.figure.savefig(title, bbox_inches='tight')


def find_windows_unique(windows, windows_str, predictions):
    '''
    Maps numeric predictions to the unique window they fall into. If a prediction falls into multiple windows, it is assigned to the last matching window.
    Should only be used with non-overlapping windows.
    '''
    indices = []
    n_nomatch =0

    for prediction in predictions:
        matching_indices = np.nan
        for i, window in enumerate(windows):
            start, end = window
            if start <= prediction <= end:
                matching_indices = i

        if np.isnan(matching_indices):
            n_nomatch +=1
        indices.append(matching_indices)

    print('missing matches: {}'.format(n_nomatch))
    indices = pd.Series(data=indices, index=predictions.index)
    time_cat = indices.apply(lambda x: windows_str[int(x)] if not pd.isna(x) else np.nan)

    return time_cat


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


def calculate_overlapping_accuracy(y_true, y_pred_overlapping):
    correct_count = 0
    total_count = len(y_true)

    for true_label, pred_string in zip(y_true, y_pred_overlapping):
        # Handle NaNs (no prediction matched any window)
        if pd.isna(pred_string):
            continue
            
        # Split the predicted string into a set of valid windows
        valid_windows = set(pred_string.split(';'))
        
        # Check if the ground truth is in the valid predictions
        if str(true_label) in valid_windows:
            correct_count += 1

    return correct_count / total_count if total_count > 0 else 0.0


def plot_psuedo_time_dist(df, method, save_loc=f"pseudo_time_dist.pdf"):
    fig, ax = plt.subplots(figsize=(417/72.27, 417/72.27 * 0.618), layout='tight')
    
    sns.kdeplot(data=df, x='pred_time_unscaled', hue='gt_time_class', fill=True, alpha=0.5, ax=ax)

    ax.set_title(f"Distribution of Pseudo-times for {method}", fontdict={"fontweight": "bold"}, fontsize=8)
    ax.set_xlabel("Pseudo-time")
    ax.set_ylabel("Density")
    fig.savefig(save_loc)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment metrics")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--records", required=True, help="path to YAML file with records")
    parser.add_argument("--metrics", required=False, help="Comma-separated list of metrics to plot", default="mse,accuracy")
    parser.add_argument("--method_colours", required=False, help="JSON string of method color indices", default=None)
    parser.add_argument("--colormap", required=False, help="Matplotlib colormap name", default='tab20')
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

    plt.style.use('default')
    EXTRA_SMALL = 6
    SMALL_SIZE = 6
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=EXTRA_SMALL)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=EXTRA_SMALL)    # fontsize of the tick labels
    plt.rc('legend', fontsize=EXTRA_SMALL)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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


    df, results = add_records_to_df(records, missing_counter, missing_records, experiment, gt_window_series=anndata.obs['windowed_day_exp1'])

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
    
    for metric in metrics:
        plot_boxplot_metric_by_sub_exp(df, experiment, methods, metric=metric, title=f"{metric}_by_col_{experiment}.pdf")
    
    plot_rank_by_column_root(df, experiment, metrics, 
                        method_color_indices=method_colours_indices,
                        colormap=args.colormap,title=f"rank_by_exp_{experiment}.pdf")

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

    for dataset in results:
        for method in results[dataset]:
            for column in results[dataset][method]:
                os.makedirs(f'psuedo_time_dist_plots/{dataset}/{method}/', exist_ok=True)
                plot_psuedo_time_dist(results[dataset][method][column], method , save_loc=f'psuedo_time_dist_plots/{dataset}/{method}/pseudo_time_dist_{column}.pdf')
                plt.close()

    if experiment in ['out_dist']:
        df_no_split, results_no_split = add_records_to_df(records, missing_counter_ns, missing_records_ns, experiment, split_col=None, gt_window_series=anndata.obs['windowed_day_exp1'])
        df_no_split['accuracy'] = df_no_split['accuracy_gt_windows']
        #print(df_no_split.head())

        # use named level for clarity
        if 'method' in df.index.names:
            methods = df.index.get_level_values('method').unique().tolist()
        else:
            methods = df.index.get_level_values(1).unique().tolist()
        metrics = ['spearman', 'kendall', 'pearson', 'mse', 'accuracy']
        for metric in metrics:
            plot_boxplot_metric_by_sub_exp(df_no_split, experiment, methods, metric=metric, title=f"{metric}_by_col_{experiment}_no_splits.pdf")
        
        plot_rank_by_column_root(df_no_split, experiment, metrics,
                                 method_color_indices=method_colours_indices,
                                 colormap=args.colormap,
                                 title=f"rank_by_exp_{experiment}_no_splits.pdf")

        print("generating plots based on extrapolation and interpolation...")
        print(df)
        split_values = df.index.get_level_values('split').unique().dropna()
        # Ensure we treat them as floats for comparison
        split_floats = split_values.astype(float)
        

        # 2. Create Boolean masks using the MultiIndex values
        idx_vals = df.index.get_level_values('split').astype(float)
        exp_vals = df.index.get_level_values('exp')
        
        is_extrap = [False] * len(df)  # Initialize with False
        for exp in exp_vals.unique():
            print(f"\nProcessing exp: {exp}")
            exp_mask = exp_vals == exp
            idx_vals_exp = idx_vals[exp_mask]
            
            min_split_exp = split_floats[exp_mask].min()
            max_split_exp = split_floats[exp_mask].max()

            print(f"Timepoints detected - Min: {min_split_exp}, Max: {max_split_exp} for exp: {exp}")

            is_extrap = ((idx_vals == min_split_exp) | (idx_vals == max_split_exp)) & exp_mask
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
                                            title=f"{metric}_by_col_{experiment}_extrapolation.pdf")
            plot_rank_by_column_root(df_extrap, experiment, metrics,
                                   method_color_indices=method_colours_indices,
                                   colormap=args.colormap,
                                   title=f"rank_by_exp_{experiment}_extrapolation.pdf")
        metrics = ['mse', 'accuracy']

        # 5. Plot Interpolation
        if not df_interp.empty:
            print(f"Plotting Interpolation ({len(df_interp)} rows)")
            for metric in metrics:
                plot_boxplot_metric_by_sub_exp(df_interp, experiment, methods, metric=metric, 
                                            title=f"{metric}_by_col_{experiment}_interpolation.pdf")
            plot_rank_by_column_root(df_interp, experiment, metrics, 
                                     method_color_indices=method_colours_indices,
                                    colormap=args.colormap,
                                    title=f"rank_by_exp_{experiment}_interpolation.pdf")

