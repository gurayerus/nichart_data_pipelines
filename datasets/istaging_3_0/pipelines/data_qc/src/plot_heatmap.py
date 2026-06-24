"""
Correlation heatmap for QC — pairwise Pearson correlations between selected columns.

Useful for:
  • Seeing how ICV-correction or residualisation changes inter-ROI correlations.
  • Spotting redundant ROIs (high r) or validating that correction removed an artefact.

col: create one heatmap panel per unique value of a column (e.g. Sex), side by side.
     Correlation is computed on the subset for each panel value.

var_y: list of columns whose pairwise correlations are shown.

name: base label used in the output filename.

Output:
  <output_dir>/heatmap/<name>.png
  <output_dir>/heatmap/<name>_col_<col>.png

Usage:
  python plot_heatmap.py \\
      --data_csv   ../../output/qc/dset_qc_v5/data.csv \\
      --var_y DLMUSE_601 DLMUSE_601_corr DLMUSE_601_resid DLMUSE_509 DLMUSE_702 \\
      --name roi_correlations \\
      --col Sex \\
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
import numpy as np
import pandas as pd
import seaborn as sns

PLOT_NAME = "heatmap"

sns.set_theme(style="white", font_scale=0.85)

_log = logging.getLogger(__name__)


def _panel_size(n_vars: int) -> tuple[float, float]:
    side = max(4.5, n_vars * 0.75)
    return side, side


def _draw_panel(ax, data: pd.DataFrame, var_y: list[str]):
    corr = data[var_y].corr()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)

    sns.heatmap(
        corr, ax=ax,
        mask=mask,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        square=True,
        annot_kws={"size": max(5, 10 - len(var_y))},
        cbar_kws={"shrink": 0.7},
    )
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0,  labelsize=8)


def plot_heatmap(
    data_csv: Path,
    var_y: list[str],
    name: str,
    output_dir: Path,
    plot_name: str = PLOT_NAME,
    col: str | None = None,
    title: str | None = None,
):
    if not data_csv.exists():
        _log.error(f"Data CSV not found: {data_csv}")
        sys.exit(1)

    df = pd.read_csv(data_csv, low_memory=False)

    missing = [c for c in var_y if c not in df.columns]
    if missing:
        _log.error(f"Columns not found in data: {missing}")
        sys.exit(1)
    if col and col not in df.columns:
        _log.error(f"Column not found in data: '{col}'")
        sys.exit(1)

    if len(var_y) < 2:
        _log.error("plot_heatmap requires at least 2 columns in var_y.")
        sys.exit(1)

    w, h = _panel_size(len(var_y))

    if col:
        col_vals = sorted(df[col].dropna().unique())
        n_panels = len(col_vals)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(w * n_panels + 0.5, h),
        )
        if n_panels == 1:
            axes = [axes]

        for ax, val in zip(axes, col_vals):
            subset = df[df[col] == val][var_y].dropna()
            _draw_panel(ax, subset, var_y)
            ax.set_title(f"{col} = {val}", fontsize=9)

        fig.suptitle(title or f"Correlation heatmap | facet: {col}", y=1.02, fontsize=10)

    else:
        fig, ax = plt.subplots(figsize=(w, h))
        _draw_panel(ax, df[var_y].dropna(), var_y)
        ax.set_title(title or "Correlation heatmap")

    fig.tight_layout()

    fname = name + (f"_col_{col}" if col else "") + ".png"
    out_path = output_dir / plot_name / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        _log.info(f"Skipping (exists): {out_path}")
        plt.close(fig)
        return

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Correlation heatmap for QC.")
    parser.add_argument("--data_csv",   required=True)
    parser.add_argument("--var_y",      required=True, nargs="+", help="columns to correlate")
    parser.add_argument("--name",       required=True, help="base label for output filename")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plot_name",  default=PLOT_NAME)
    parser.add_argument("--col",        default=None, help="column to facet into side-by-side panels")
    parser.add_argument("--title",   default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log_dir", default=None)
    args = parser.parse_args()

    setup_logger(__name__, verbose=args.verbose,
                 log_dir=Path(args.log_dir) if args.log_dir else None)

    plot_heatmap(
        data_csv=Path(args.data_csv),
        var_y=args.var_y,
        name=args.name,
        output_dir=Path(args.output_dir),
        plot_name=args.plot_name,
        col=args.col,
        title=args.title,
    )


if __name__ == "__main__":
    main()
