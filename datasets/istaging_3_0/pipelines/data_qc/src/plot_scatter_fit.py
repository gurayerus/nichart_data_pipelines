"""
Scatter plot with optional fit line for QC.

fit_type controls the fit:
  (omitted)  — scatter only, no fit
  linear     — linear regression + 95% CI; annotates R², slope, p-value
  lowess     — LOWESS smooth curve

hue: color points and draw per-group fits; compound "A+B" merges columns.
col: facet into one subplot panel per unique value of the column.

Output:
  <output_dir>/<plot_name>/<var_y>_<var_x>.png
  <output_dir>/<plot_name>/<var_y>_<var_x>_by_<hue>.png
  <output_dir>/<plot_name>/<var_y>_<var_x>_col_<col>.png
  <output_dir>/<plot_name>/<var_y>_<var_x>_by_<hue>_col_<col>.png

Usage:
  python plot_scatter_fit.py \\
      --data_csv   ../../output/qc/dset_qc_v1/data.csv \\
      --var_x      Age --var_y DLMUSE_601 \\
      --fit_type   lowess --hue Study --col Sex \\
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
from scipy import stats

PLOT_NAME = "scatter_fit"

sns.set_theme(style="whitegrid", font_scale=0.9)

_log = logging.getLogger(__name__)


def _regress(x: pd.Series, y: pd.Series) -> tuple[float, float, str]:
    slope, _, r, p_value, _ = stats.linregress(x, y)
    p_label = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    return r ** 2, slope, p_label


def _draw_fit(ax, data: pd.DataFrame, var_x: str, var_y: str,
              fit_type: str, color, linewidth: float,
              linestyle: str = "-", ci: int | None = 95):
    if fit_type == "linear":
        sns.regplot(
            data=data, x=var_x, y=var_y, scatter=False,
            line_kws={"color": color, "linewidth": linewidth, "linestyle": linestyle},
            ci=ci, ax=ax,
        )
    elif fit_type == "lowess":
        sns.regplot(
            data=data, x=var_x, y=var_y, scatter=False,
            lowess=True,
            line_kws={"color": color, "linewidth": linewidth, "linestyle": linestyle},
            ax=ax,
        )


def _draw_panel(ax, data: pd.DataFrame, var_x: str, var_y: str,
                fit_type: str | None, hue: str | None,
                palette: list | None = None):
    """Draw scatter + fits on a single axes panel."""
    if hue:
        groups = sorted(data[hue].dropna().unique())
        pal    = palette or sns.color_palette(
            "tab10" if len(groups) <= 10 else "tab20", n_colors=len(groups)
        )
        for grp, color in zip(groups, pal):
            sub = data[data[hue] == grp]
            ax.scatter(sub[var_x], sub[var_y], s=5, alpha=0.25, color=color, label=str(grp))
            if fit_type and len(sub) >= 3:
                _draw_fit(ax, sub, var_x, var_y, fit_type,
                          color=color, linewidth=1.2, ci=95)
        if fit_type:
            _draw_fit(ax, data, var_x, var_y, fit_type,
                      color="black", linewidth=2, linestyle="--", ci=None)
    else:
        ax.scatter(data[var_x], data[var_y], s=5, alpha=0.3, color="steelblue")
        if fit_type:
            _draw_fit(ax, data, var_x, var_y, fit_type,
                      color="crimson", linewidth=1.5, ci=95)

    # annotation
    if fit_type == "linear" and len(data) >= 3:
        r2, slope, p_label = _regress(data[var_x], data[var_y])
        annot = f"R² = {r2:.3f}   {p_label}   slope = {slope:.3g}"
    else:
        annot = f"N = {len(data)}"
    ax.annotate(
        annot, xy=(0.03, 0.96), xycoords="axes fraction",
        fontsize=7, va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.7},
    )
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)


def plot_scatter_fit(
    data_csv: Path,
    var_x: str,
    var_y: str,
    output_dir: Path,
    plot_name: str = PLOT_NAME,
    fit_type: str | None = None,
    hue: str | None = None,
    col: str | None = None,
    title: str | None = None,
):
    if not data_csv.exists():
        _log.error(f"Data CSV not found: {data_csv}")
        sys.exit(1)

    if fit_type not in (None, "linear", "lowess"):
        _log.error(f"Unknown fit_type '{fit_type}'. Choose: linear, lowess, or omit.")
        sys.exit(1)

    df = pd.read_csv(data_csv, low_memory=False)

    hue_parts = [p.strip() for p in hue.split("+")] if hue else []
    for c in [var_x, var_y] + hue_parts + ([col] if col else []):
        if c not in df.columns:
            _log.error(f"Column not found in data: '{c}'")
            sys.exit(1)

    keep = [var_x, var_y] + hue_parts + ([col] if col else [])
    data = df[keep].dropna()

    # build combined hue column for compound keys like "Study+Sex"
    if len(hue_parts) > 1:
        data = data.copy()
        data[hue] = data[hue_parts[0]].astype(str)
        for part in hue_parts[1:]:
            data[hue] = data[hue] + " / " + data[part].astype(str)

    if len(data) < 3:
        _log.error(f"Not enough data after dropping NaNs ({len(data)} rows).")
        sys.exit(1)

    # consistent hue palette across all panels
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
            figsize=(5 * n_panels, 5),
            sharey=True, sharex=True,
        )
        if n_panels == 1:
            axes = [axes]

        for ax, val in zip(axes, col_vals):
            subset = data[data[col] == val]
            _draw_panel(ax, subset, var_x, var_y, fit_type, hue, palette)
            ax.set_title(f"{col} = {val}", fontsize=9)
            if ax is not axes[0]:
                ax.set_ylabel("")

        # single legend from last panel
        if hue:
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(
                handles, labels, title=hue, fontsize=7,
                bbox_to_anchor=(1.01, 0.9), loc="upper left",
            )

        suptitle = title or (
            f"{var_y} vs {var_x}"
            + (f" by {hue}" if hue else "")
            + f" | facet: {col}"
        )
        fig.suptitle(suptitle, y=1.02, fontsize=10)

    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_panel(ax, data, var_x, var_y, fit_type, hue, palette)
        ax.set_title(title or (f"{var_y} vs {var_x} by {hue}" if hue else f"{var_y} vs {var_x}"))

        if hue:
            ax.legend(
                title=hue, fontsize=7, markerscale=2,
                bbox_to_anchor=(1.01, 1), loc="upper left",
            )

    fig.tight_layout()

    # ── output path ───────────────────────────────────────────────────────────
    parts = [var_x]
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
        description="Scatter plot with optional fit line for QC."
    )
    parser.add_argument("--data_csv",   required=True, help="input data CSV")
    parser.add_argument("--var_x",      required=True, help="x-axis column name")
    parser.add_argument("--var_y",      required=True, help="y-axis column name")
    parser.add_argument("--output_dir", required=True,
                        help="base output directory")
    parser.add_argument("--plot_name",  default=PLOT_NAME,
                        help=f"sub-directory name for this plot type (default: {PLOT_NAME})")
    parser.add_argument("--fit_type",   default=None, choices=["linear", "lowess"],
                        help="fit line type: linear or lowess; omit for scatter only")
    parser.add_argument("--hue",        default=None,
                        help="column to color/group by; compound: 'Study+Sex'")
    parser.add_argument("--col",        default=None,
                        help="column to facet into separate panels (one panel per unique value)")
    parser.add_argument("--title",      default=None, help="optional plot title override")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="show DEBUG messages on console")
    parser.add_argument("--log_dir",  default=None,
                        help="directory for WARNING/ERROR log file")
    args = parser.parse_args()

    setup_logger(__name__, verbose=args.verbose,
                 log_dir=Path(args.log_dir) if args.log_dir else None)

    plot_scatter_fit(
        data_csv=Path(args.data_csv),
        var_x=args.var_x,
        var_y=args.var_y,
        output_dir=Path(args.output_dir),
        plot_name=args.plot_name,
        fit_type=args.fit_type,
        hue=args.hue,
        col=args.col,
        title=args.title,
    )


if __name__ == "__main__":
    main()
