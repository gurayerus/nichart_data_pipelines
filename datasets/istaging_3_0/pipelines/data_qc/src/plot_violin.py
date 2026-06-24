"""
Violin plot for QC — distribution shape + quartiles per group.

Combines what a box plot and a KDE each show separately.  Useful wherever
box plots are used; particularly informative when a group has a bimodal or
skewed distribution that a box plot would hide.

col: facet into one panel per unique value of a column (e.g. Sex).

Output:
  <output_dir>/violin/<var_y>_<var_x>.png
  <output_dir>/violin/<var_y>_<var_x>_col_<col>.png

Usage:
  python plot_violin.py \\
      --data_csv   ../../output/qc/dset_qc_v5/data.csv \\
      --var_x      Study --var_y DLMUSE_601 \\
      --col        Sex \\
      --output_dir ../../output/qc/dset_qc_v5/results
"""

import argparse
import logging
import sys
from pathlib import Path

_PIPELINES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PIPELINES / "utils"))
from logger import setup_logger  # type: ignore[import]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PLOT_NAME = "violin"

sns.set_theme(style="whitegrid", font_scale=0.9)

_log = logging.getLogger(__name__)


def _fig_width(n_cats: int) -> float:
    return max(8.0, n_cats * 0.55)


def _draw_panel(ax, data: pd.DataFrame, var_x: str, var_y: str, order: list):
    sns.violinplot(
        data=data, x=var_x, y=var_y, order=order, ax=ax,
        inner="box", linewidth=0.7, cut=0,
    )

    n_cats = len(order)
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)

    if n_cats > 6:
        ax.tick_params(axis="x", labelrotation=90,
                       labelsize=max(4, 9 - n_cats // 8))

    counts = data.groupby(var_x)[var_y].count()
    for i, cat in enumerate(order):
        n = counts.get(cat, 0)
        ax.text(
            i, ax.get_ylim()[0], f"n={n}",
            ha="center", va="top", fontsize=5, color="dimgrey",
        )


def plot_violin(
    data_csv: Path,
    var_x: str,
    var_y: str,
    output_dir: Path,
    plot_name: str = PLOT_NAME,
    col: str | None = None,
    title: str | None = None,
):
    if not data_csv.exists():
        _log.error(f"Data CSV not found: {data_csv}")
        sys.exit(1)

    df = pd.read_csv(data_csv, low_memory=False)

    for c in [var_x, var_y] + ([col] if col else []):
        if c not in df.columns:
            _log.error(f"Column not found in data: '{c}'")
            sys.exit(1)

    keep  = [var_x, var_y] + ([col] if col else [])
    data  = df[keep].dropna(subset=[var_x, var_y])
    order = sorted(data[var_x].unique())

    if col:
        col_vals = sorted(data[col].dropna().unique())
        n_panels = len(col_vals)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(_fig_width(len(order)), 5),
            sharey=True, sharex=True,
        )
        if n_panels == 1:
            axes = [axes]

        for ax, val in zip(axes, col_vals):
            subset = data[data[col] == val]
            _draw_panel(ax, subset, var_x, var_y, order)
            ax.set_title(f"{col} = {val}", fontsize=9)
            if ax is not axes[0]:
                ax.set_ylabel("")

        fig.suptitle(title or f"{var_y} by {var_x} | facet: {col}", y=1.02, fontsize=10)

    else:
        fig, ax = plt.subplots(figsize=(_fig_width(len(order)), 5))
        _draw_panel(ax, data, var_x, var_y, order)
        ax.set_title(title or f"{var_y} by {var_x}")

    fig.tight_layout()

    parts = [var_x] + ([f"col_{col}"] if col else [])
    out_path = output_dir / plot_name / (var_y + "_" + "_".join(parts) + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        _log.info(f"Skipping (exists): {out_path}")
        plt.close(fig)
        return

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Violin plot for QC.")
    parser.add_argument("--data_csv",   required=True)
    parser.add_argument("--var_x",      required=True, help="categorical grouping column")
    parser.add_argument("--var_y",      required=True, help="continuous column to plot")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plot_name",  default=PLOT_NAME)
    parser.add_argument("--col",        default=None)
    parser.add_argument("--title",   default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log_dir", default=None)
    args = parser.parse_args()

    setup_logger(__name__, verbose=args.verbose,
                 log_dir=Path(args.log_dir) if args.log_dir else None)

    plot_violin(
        data_csv=Path(args.data_csv),
        var_x=args.var_x,
        var_y=args.var_y,
        output_dir=Path(args.output_dir),
        plot_name=args.plot_name,
        col=args.col,
        title=args.title,
    )


if __name__ == "__main__":
    main()
