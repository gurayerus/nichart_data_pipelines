"""
Distribution (KDE density) plot for QC.

Shows the density of var_y, optionally split by a hue column and/or faceted
into one panel per value of a col column.  Mirrors the interface of
plot_scatter_fit.py so the same var_y / hue / col lists from data_desc.json
can be reused for both plot types.

Output:
  <output_dir>/dist/<var_y>_dist.png
  <output_dir>/dist/<var_y>_dist_by_<hue>.png
  <output_dir>/dist/<var_y>_dist_col_<col>.png
  <output_dir>/dist/<var_y>_dist_by_<hue>_col_<col>.png

Usage:
  python plot_dist.py \\
      --data_csv   ../../output/qc/dset_qc_v1/data.csv \\
      --var_y      DLMUSE_601 \\
      --hue        Study-study2 --col Sex \\
      --output_dir ../../output/qc/dset_qc_v1/results
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

PLOT_NAME = "dist"

sns.set_theme(style="whitegrid", font_scale=0.9)

_log = logging.getLogger(__name__)


def _draw_panel(ax, data: pd.DataFrame, var_y: str, hue: str | None,
                palette: list | None = None):
    """Draw KDE curves on a single axes panel."""
    if hue:
        groups = sorted(data[hue].dropna().unique())
        pal    = palette or sns.color_palette(
            "tab10" if len(groups) <= 10 else "tab20", n_colors=len(groups)
        )
        for grp, color in zip(groups, pal):
            sub = data.loc[data[hue] == grp, var_y].dropna()
            if len(sub) < 2:
                continue
            sns.kdeplot(sub, ax=ax, label=str(grp), color=color, alpha=0.7, linewidth=1.4)
        # overall distribution in black
        sns.kdeplot(data[var_y].dropna(), ax=ax, color="black",
                    linewidth=1.5, linestyle="--", label="overall")
    else:
        sns.kdeplot(data[var_y].dropna(), ax=ax, color="steelblue",
                    linewidth=1.5, fill=True, alpha=0.2)

    ax.set_xlabel(var_y)
    ax.set_ylabel("Density")
    n = data[var_y].notna().sum()
    ax.annotate(
        f"N = {n}", xy=(0.97, 0.96), xycoords="axes fraction",
        fontsize=7, ha="right", va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.7},
    )


def plot_dist(
    data_csv: Path,
    var_y: str,
    output_dir: Path,
    plot_name: str = PLOT_NAME,
    hue: str | None = None,
    col: str | None = None,
    title: str | None = None,
):
    if not data_csv.exists():
        _log.error(f"Data CSV not found: {data_csv}")
        sys.exit(1)

    df = pd.read_csv(data_csv, low_memory=False)

    hue_parts = [p.strip() for p in hue.split("+")] if hue else []
    for c in [var_y] + hue_parts + ([col] if col else []):
        if c not in df.columns:
            _log.error(f"Column not found in data: '{c}'")
            sys.exit(1)

    keep = [var_y] + hue_parts + ([col] if col else [])
    data = df[keep].dropna(subset=[var_y])

    # build combined hue column for compound keys like "Study+Sex"
    if len(hue_parts) > 1:
        data = data.copy()
        data[hue] = data[hue_parts[0]].astype(str)
        for part in hue_parts[1:]:
            data[hue] = data[hue] + " / " + data[part].astype(str)

    if data[var_y].notna().sum() < 2:
        _log.error(f"Not enough data for '{var_y}' ({data[var_y].notna().sum()} valid rows).")
        sys.exit(1)

    palette = None
    if hue:
        groups  = sorted(data[hue].dropna().unique())
        palette = sns.color_palette(
            "tab10" if len(groups) <= 10 else "tab20", n_colors=len(groups)
        )

    # ── facet panels ──────────────────────────────────────────────────────────
    if col:
        col_vals = sorted(data[col].dropna().unique())
        n_panels = len(col_vals)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(5 * n_panels, 4),
            sharey=True, sharex=True,
        )
        if n_panels == 1:
            axes = [axes]

        for ax, val in zip(axes, col_vals):
            subset = data[data[col] == val]
            _draw_panel(ax, subset, var_y, hue, palette)
            ax.set_title(f"{col} = {val}", fontsize=9)
            if ax is not axes[0]:
                ax.set_ylabel("")

        if hue:
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(
                handles, labels, title=hue, fontsize=7,
                bbox_to_anchor=(1.01, 0.9), loc="upper left",
            )

        suptitle = title or (
            f"Distribution of {var_y}"
            + (f" by {hue}" if hue else "")
            + f" | facet: {col}"
        )
        fig.suptitle(suptitle, y=1.02, fontsize=10)

    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        _draw_panel(ax, data, var_y, hue, palette)
        ax.set_title(title or (
            f"Distribution of {var_y}" + (f" by {hue}" if hue else "")
        ))
        if hue:
            ax.legend(title=hue, fontsize=7, markerscale=1.5,
                      bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.tight_layout()

    # ── output path ───────────────────────────────────────────────────────────
    parts = ["dist"]
    if hue:
        parts += [f"by_{hue}"]
    if col:
        parts += [f"col_{col}"]
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
    parser = argparse.ArgumentParser(
        description="Distribution (KDE) plot for QC."
    )
    parser.add_argument("--data_csv",   required=True, help="input data CSV")
    parser.add_argument("--var_y",      required=True, help="column whose distribution to plot")
    parser.add_argument("--output_dir", required=True, help="base output directory")
    parser.add_argument("--plot_name",  default=PLOT_NAME,
                        help=f"sub-directory name for this plot type (default: {PLOT_NAME})")
    parser.add_argument("--hue",   default=None,
                        help="column to color/group by; compound: 'Study+Sex'")
    parser.add_argument("--col",   default=None,
                        help="column to facet into separate panels")
    parser.add_argument("--title",   default=None, help="optional plot title override")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log_dir", default=None)
    args = parser.parse_args()

    setup_logger(__name__, verbose=args.verbose,
                 log_dir=Path(args.log_dir) if args.log_dir else None)

    plot_dist(
        data_csv=Path(args.data_csv),
        var_y=args.var_y,
        output_dir=Path(args.output_dir),
        plot_name=args.plot_name,
        hue=args.hue,
        col=args.col,
        title=args.title,
    )


if __name__ == "__main__":
    main()
