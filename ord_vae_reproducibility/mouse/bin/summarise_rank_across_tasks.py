#!/usr/bin/env python3
"""Create a compact cross-task ranking summary figure and tables.

This script condenses model rankings across:
- tasks: train, test-in, test-out
- train labels: base, coarsened, overlapping

It computes average method rank per condition using a common metric set
(default: mse and accuracy), then reports winners and a global leaderboard.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append("/mnt/mr01-data01/j72687wm/ordVae_paper/scripts/")
from analyse_experiments_utils import *


TASK_PATHS = {
    "train": "results/analysis/test_train/train_plots/combined_metrics_test_train_train.csv",
    "test-in": "results/analysis/test_train/test_indist_plot/combined_metrics_test_train_test_in_dist.csv",
    "test-out": "results/analysis/test_train/test_interpolation_plot/combined_metrics_test_train_test_interpol.csv",
}

EXP_LABELS = {
    "windowed_day_base": "base",
    "windowed_day_coarsened": "coarsened",
    "windowed_day_overlapping": "overlapping",
}

POSITIVE_METRICS = {"spearman", "kendall", "pearson", "accuracy", "accuracy_gt_windows", "accuracy_train_windows"}
NEGATIVE_METRICS = {"mse"}

METHOD_SHORT = {
    "sceptic": "sceptic",
    "ga_reg": "ga_reg",
    "ga_ord": "ga_ord",
    "ga_cumulative_logit": "ga_cl",
    "nb_reg": "nb_reg",
    "nb_ord": "nb_ord",
    "nb_cumulative_logit": "nb_cl",
    "pypsupertime": "psuper",
    "temporalVAE": "tempVAE",
}

FONT_SIZE_TITLE = 8
FONT_SIZE_LABEL = 6
FONT_SIZE_TICK = 6
FONT_SIZE_LEGEND = 6
TEXT_WIDTH = 526


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise model rankings across tasks and train-label schemes")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mse", "accuracy", "spearman", "kendall", "pearson"],
        help="Metrics used to build comparable cross-task ranks (default: mse accuracy spearman kendall pearson)",
    )
    parser.add_argument("--outdir", type=Path, default=Path("results/analysis/test_train/summary"), help="Output directory")
    return parser.parse_args()


def metric_direction(metric: str) -> bool:
    """Return ascending flag for rank() where True means lower is better."""
    if metric in NEGATIVE_METRICS:
        return True
    if metric in POSITIVE_METRICS:
        return False
    raise ValueError(f"Unknown direction for metric '{metric}'")


def load_task_df(root: Path, task: str) -> pd.DataFrame:
    path = root / TASK_PATHS[task]
    if not path.exists():
        raise FileNotFoundError(f"Missing input file for task '{task}': {path}")

    df = pd.read_csv(path)
    required = {"dataset", "method", "exp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input file {path} is missing required columns: {sorted(missing)}")

    df["task"] = task
    df["label"] = df["exp"].map(EXP_LABELS)
    return df


def ranks_by_condition(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for (task, label), sdf in df.groupby(["task", "label"], dropna=False):
        if pd.isna(label):
            continue

        metric_rank_series: Dict[str, pd.Series] = {}

        for metric in metrics:
            if metric not in sdf.columns:
                continue

            t = sdf[["dataset", "method", metric]].dropna(subset=[metric]).copy()
            if t.empty:
                continue

            ascending = metric_direction(metric)
            t["rank"] = t.groupby(["dataset"])[metric].rank(ascending=ascending, method="average")
            metric_rank_series[metric] = t.groupby("method")["rank"].mean()

        if not metric_rank_series:
            continue

        rank_df = pd.DataFrame(metric_rank_series)
        rank_df["condition_rank"] = rank_df.mean(axis=1)
        rank_df["task"] = task
        rank_df["label"] = label
        rank_df["metrics_used"] = len(metric_rank_series)
        rank_df = rank_df.reset_index().rename(columns={"index": "method"})
        rows.append(rank_df)

    if not rows:
        raise ValueError("No condition ranks were computed; check metrics and input files.")

    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"method": "method"})
    return out


def ordinal_rank_table(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Return a wide table of ordinal ranks indexed by label, task, and metric."""
    methods = sorted(df["method"].dropna().unique())
    rows = []
    index = []

    for (task, label), sdf in df.groupby(["task", "label"], dropna=False):
        if pd.isna(label):
            continue

        for metric in metrics:
            if metric not in sdf.columns:
                continue

            t = sdf[["dataset", "method", metric]].dropna(subset=[metric]).copy()
            if t.empty:
                continue

            ascending = metric_direction(metric)
            t = t.sort_values(
                ["dataset", metric, "method"],
                ascending=[True, ascending, True],
                kind="mergesort",
            )
            t["ordinal_rank"] = t.groupby(["dataset"]).cumcount() + 1

            row = t.groupby("method")["ordinal_rank"].mean()
            rows.append(row)
            index.append((label, task, metric))

    if not rows:
        raise ValueError("No ordinal rank table could be built; check metrics and input files.")

    rank_table = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["label", "task", "metric"]))
    rank_table = rank_table.reindex(columns=methods)
    return rank_table.sort_index()


def borda_from_ranks(rank_table: pd.DataFrame) -> pd.DataFrame:
    """Convert ordinal ranks to Borda points for each row."""
    counts = rank_table.notna().sum(axis=1)
    return rank_table.rsub(counts, axis=0).add(1)


def summarize_borda_by_axis(borda_table: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Aggregate Borda points over one index axis and preserve method columns."""
    if axis not in borda_table.index.names:
        raise ValueError(f"Axis '{axis}' not found in Borda table index {borda_table.index.names}")

    return borda_table.groupby(level=axis).sum(numeric_only=True)


def plot_borda_panels(summary: pd.DataFrame, outpath: Path, title: str, xlabel: str = "Borda points") -> None:
    if summary.empty:
        raise ValueError(f"Cannot plot empty Borda summary for {outpath}")

    panels = list(summary.index)
    n_panels = len(panels)
    fig_height = max(1.4, 1.25 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(6.0, fig_height), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        series = summary.loc[panel].dropna().sort_values(ascending=True)
        y = np.arange(len(series))
        ax.barh(y, series.values, color="#4C78A8")
        ax.set_yticks(y, [METHOD_SHORT.get(method, method) for method in series.index])
        ax.set_title(str(panel))
        ax.set_xlabel(xlabel)

    fig.suptitle(title)
    fig.savefig(outpath, dpi=200, transparent=True)
    plt.close(fig)


def save_borda_outputs(df: pd.DataFrame, metrics: List[str], outdir: Path) -> None:
    """Build and save ordinal-rank and Borda summaries for additional views."""
    rank_table = ordinal_rank_table(df, metrics)
    rank_table.to_csv(outdir / "ordinal_ranks_label_task_metric.csv")

    borda_table = borda_from_ranks(rank_table)
    borda_table.to_csv(outdir / "borda_points_label_task_metric.csv")

    metric_summary = summarize_borda_by_axis(borda_table, "metric")
    metric_summary.loc["all_metrics"] = borda_table.sum(axis=0, numeric_only=True)
    metric_summary.to_csv(outdir / "borda_metric_summary.csv")
    plot_borda_panels(metric_summary, outdir / "borda_metric_summary.pdf", "Metric-level Borda")

    label_summary = summarize_borda_by_axis(borda_table, "label")
    label_summary.to_csv(outdir / "borda_label_summary.csv")
    plot_borda_panels(label_summary, outdir / "borda_label_summary.pdf", "Label-level Borda")

    task_summary = summarize_borda_by_axis(borda_table, "task")
    task_summary.to_csv(outdir / "borda_train_summary.csv")
    plot_borda_panels(task_summary, outdir / "borda_train_summary.pdf", "Train-level Borda", xlabel="Borda points")


def winner_table(condition_ranks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, label), sdf in condition_ranks.groupby(["task", "label"], dropna=False):
        if sdf.empty:
            continue

        s = sdf.sort_values(["condition_rank", "method"]).reset_index(drop=True)
        top = s.iloc[0]
        second = s.iloc[1] if len(s) > 1 else None

        rows.append(
            {
                "task": task,
                "label": label,
                "winner": top["method"],
                "winner_short": METHOD_SHORT.get(top["method"], top["method"]),
                "winner_rank": float(top["condition_rank"]),
                "second": None if second is None else second["method"],
                "second_rank": None if second is None else float(second["condition_rank"]),
                "gap_to_second": None if second is None else float(second["condition_rank"] - top["condition_rank"]),
            }
        )

    return pd.DataFrame(rows)


def leaderboard(condition_ranks: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    scored = condition_ranks.copy()
    scored["ordinal_rank"] = scored.groupby(["task", "label"])["condition_rank"].rank(method="average", ascending=True)
    counts = scored.groupby(["task", "label"])["method"].nunique().rename("n_methods")
    scored = scored.join(counts, on=["task", "label"])
    scored["borda_points"] = scored["n_methods"] - scored["ordinal_rank"] + 1

    wins = {method: 0 for method in scored["method"].unique()}

    for metric in metrics:
        if metric not in scored.columns:
            continue
        scored[f"{metric}_win"] = scored.groupby(["task", "label"])[metric].rank(method="average", ascending=True)
        n_wins = scored.groupby("method")[f"{metric}_win"].apply(lambda x: int(np.sum(np.isclose(x, 1.0))))

        wins.update({method: wins.get(method, 0) + n for method, n in n_wins.items()})

    agg = (
        scored.groupby("method")
        .agg(
            total_borda=("borda_points", "sum"),
            wins_task_label=("ordinal_rank", lambda x: int(np.sum(np.isclose(x, 1.0)))),
            mean_condition_rank=("condition_rank", "mean"),
            std_condition_rank=("condition_rank", "std"),
            n_conditions=("condition_rank", "count"),
        )
        .reset_index()
    )

    agg["wins_task_label_metric"] = agg["method"].map(lambda m: wins.get(m, 0))

    agg["method_short"] = agg["method"].map(lambda x: METHOD_SHORT.get(x, x))
    agg = agg.sort_values(["total_borda", "wins_task_label", "mean_condition_rank"], ascending=[False, False, True]).reset_index(drop=True)
    return agg


def plot_summary(winners: pd.DataFrame, board: pd.DataFrame, outpath: Path) -> None:
    task_order = ["train", "test-in", "test-out"]
    label_order = ["base", "coarsened", "overlapping"]

    values = np.full((len(task_order), len(label_order)), np.nan)
    text = [["" for _ in label_order] for _ in task_order]

    for _, row in winners.iterrows():
        i = task_order.index(row["task"])
        j = label_order.index(row["label"])
        values[i, j] = 0.0
        gap = row["gap_to_second"]
        gap_str = "na" if pd.isna(gap) else f"{gap:.2f}"
        text[i][j] = f"{row['winner_short']}\nr={row['winner_rank']:.2f}\nd={gap_str}"

    rank_fig_size = (0.75 * TEXT_WIDTH / 72.27, (0.75 * 0.618 * 0.5 * TEXT_WIDTH) / 72.27)
    fig = plt.figure(figsize=rank_fig_size, layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(values, aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax0.text(j, i, text[i][j], ha="center", va="center", fontsize=FONT_SIZE_LABEL)

    ax0.set_xticks(np.arange(len(label_order)), label_order)
    ax0.set_yticks(np.arange(len(task_order)), task_order)
    ax0.set_title("Condition Winners")

    ax1 = fig.add_subplot(gs[0, 1])
    top = board.sort_values("total_borda", ascending=True)
    y = np.arange(len(top))
    ax1.barh(y, top["total_borda"], color="#4C78A8")
    ax1.set_yticks(y, top["method_short"])
    ax1.set_xlabel("Total Borda points")
    ax1.set_title("Dataset Leaderboard")

    for yi, (_, row) in enumerate(top.iterrows()):
        if row["total_borda"] > 20:
            ax1.text(0.05, yi, f"wins={int(row['wins_task_label_metric'])}", va="center", fontsize=FONT_SIZE_LABEL, color="white")
        else:
            ax1.text(row["total_borda"] + 0.5, yi, f"wins={int(row['wins_task_label_metric'])}", va="center", fontsize=FONT_SIZE_LABEL)

    fig.savefig(outpath, dpi=200, transparent=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    set_publication_style()

    df_all = []
    for task in ["train", "test-in", "test-out"]:
        df_all.append(load_task_df(args.root, task))
    df = pd.concat(df_all, ignore_index=True)

    unknown = [m for m in args.metrics if m not in POSITIVE_METRICS and m not in NEGATIVE_METRICS]
    if unknown:
        raise ValueError(
            "Unsupported metrics for ranking direction. "
            f"Add them to POSITIVE_METRICS/NEGATIVE_METRICS in script. Unknown: {unknown}"
        )

    condition_ranks = ranks_by_condition(df, args.metrics)
    winners = winner_table(condition_ranks)
    board = leaderboard(condition_ranks, args.metrics)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    condition_ranks.to_csv(outdir / "condition_ranks.csv", index=False)
    winners.to_csv(outdir / "condition_winners.csv", index=False)
    board.to_csv(outdir / "leaderboard.csv", index=False)

    save_borda_outputs(df, args.metrics, outdir)

    plot_summary(winners, board, outdir / "rank_summary_across_tasks.pdf")

    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()