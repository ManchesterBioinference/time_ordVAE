import sys

import scanpy as sc
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
sys.path.append("/mnt/mr01-data01/j72687wm/ordVae_paper/scripts/")
from analyse_experiments_utils import set_publication_style

FONT_SIZE_TITLE = 8
FONT_SIZE_LABEL = 7
FONT_SIZE_TICK = 6
FONT_SIZE_LEGEND = 6
TEXT_WIDTH = 526

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data summary figure from processed h5ad file.')
    parser.add_argument('--h5ad_file', type=str, help='Path to the processed h5ad file.')
    parser.add_argument('--column_windows', type=str, help='comma seperated list of column names to create windows for.')
    parser.add_argument('--out_file', type=str, default='data_summary.png', help='Path to save the output summary figure.')
    args = parser.parse_args()

    set_publication_style()

    # Load the processed h5ad file
    adata = sc.read_h5ad(args.h5ad_file)
    # Generate summary statistics
    num_cells = adata.n_obs
    num_genes = adata.n_vars

    window_info = {}

    for col in args.column_windows.split(','):
        if col in adata.obs:
            window_info[col] = {
                'num_windows': adata.obs[col].nunique(),
                'window_ranges': sorted(adata.obs[col].unique().tolist())
            }

    day_num_vc = adata.obs['day_num'].value_counts().sort_index()
    day_vc = adata.obs['day'].value_counts().sort_index()

    with open('window_info.txt', 'w') as f:
        print(f'Number of cells: {num_cells}', file=f)
        print(f'Number of genes: {num_genes}', file=f)
        print(window_info, file=f)
        print(f'\n\nDay Number Value Counts:\n{day_num_vc}\n\nDay Value Counts:\n{day_vc}', file=f)

    edges_vc = {}
    edges_numeric = {}

    exps_short = {
        'windowed_day_base': 'base',
        'windowed_day_coarsened': 'coarsened',
        'windowed_day_overlapping': 'overlapping'
    }

    for col in args.column_windows.split(','):
        if col in adata.obs:
            plot_col = exps_short.get(col, col)
            edges_vc[plot_col] = adata.obs[col].value_counts().sort_index()
            str_edges = sorted(adata.obs[col].unique().tolist())
            numeric_edges = np.array([np.array(el.split('_'), dtype=float) for el in str_edges])
            numeric_edges.sort(axis=0)
            edges_numeric[plot_col] = numeric_edges
    # Create summary figure

    print(edges_vc)


    fig, ax= plt.subplots(figsize=((0.25*TEXT_WIDTH/72.27),(0.75*0.618*0.5*TEXT_WIDTH) / 72.27), tight_layout=True)

    jitter = 1/6
    plotted_y = []
    ys = np.arange(max([edge.shape[0] for edge in edges_numeric.values()]), 0, -1)

    print(ys)
    for i, (name, edge) in enumerate(edges_numeric.items()):

        ys_plot = ys + (jitter * i)

        if ys_plot.shape[0] != edge.shape[0]:
            ys_plot = ys_plot[:edge.shape[0]]
            
        ax.hlines(ys_plot, edge[:, 0], edge[:, 1], label=name, color=plt.get_cmap('tab10')([i]))

    ax.set_title('Mouse Windows')
    ax.set_xlabel('Time')
    ax.legend(loc='lower left')
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5)
    ax.set_ylim([30, 44])
    ax.set_xlim([8, 13])
    ax.set_yticks([])

    fig.savefig(args.out_file, dpi=300)