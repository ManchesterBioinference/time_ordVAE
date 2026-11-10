import scanpy as sc
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import sys

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
    parser.add_argument('--column_window', type=str, help='column name containing time window')
    parser.add_argument('--out_file', type=str, default='data_summary.png', help='Path to save the output summary figure.')
    args = parser.parse_args()


    # Load the processed h5ad file
    adata = sc.read_h5ad(args.h5ad_file)
    # Generate summary statistics
    num_cells = adata.n_obs
    num_genes = adata.n_vars
    set_publication_style()

    window_info = {}

    for col in args.column_window.split(','):
        if col in adata.obs:
            window_info[col] = {
                'num_windows': adata.obs[col].nunique(),
                'window_ranges': sorted(adata.obs[col].unique().tolist())
            }

    time_vc = adata.obs['time'].value_counts().sort_index()
    time_window_vc = adata.obs['time_windows'].value_counts().sort_index()

    with open('window_info.txt', 'w') as f:
        print(f"Number of cells: {num_cells}", file=f)
        print(f"Number of genes: {num_genes}", file=f)
        print(window_info, file=f)
        print(f"Time value counts:\n{time_vc}", file=f)
        print(f"Time window value counts:\n{time_window_vc}", file=f)




    edges_vc = {}
    edges_numeric = {}

    
    if args.column_window in adata.obs:
        col = args.column_window
        edges_vc = adata.obs[col].value_counts().sort_index()
        str_edges = sorted(adata.obs[col].unique().tolist())
        numeric_edges = np.array([np.array(el.split('_'), dtype=float) for el in str_edges])
        numeric_edges.sort(axis=0)
        edges_numeric = numeric_edges
    else:
        raise ValueError(f"Column {args.column_window} not found in adata.obs")
    # Create summary figure

    print(edges_vc)

    fig, ax= plt.subplots(figsize=((0.25*TEXT_WIDTH/72.27),(0.75*0.618*0.5*TEXT_WIDTH) / 72.27), tight_layout=True)

    jitter = 1/6
    ys = np.arange(edges_numeric.shape[0], 0, -1)
            
    ax.hlines(ys, edges_numeric[:, 0], edges_numeric[:, 1], label=args.column_window, color=plt.get_cmap('tab10')(0))

    ax.set_title('Furlong Windows')
    ax.set_xlabel('Time')
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5)
    ax.set_yticks([])

    fig.savefig(args.out_file, dpi=300)