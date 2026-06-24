"""
PCA scatter plot for QC — projects ROI volumes onto first two principal components.

The most powerful batch-effect check available: collapses all selected ROI features
to 2D, then colours each point by a metadata variable (Study, Sex, Age).  Clusters or
gradients in the scatter reveal systematic variance from batch, site or demographic
confounds that survive after correction.

PCA is fit on ALL complete rows (no col-split), then the scatter is optionally faceted
by a column so the same projection is comparable across subgroups.

var_y: columns used as PCA input features (e.g. a set of ROI volumes).
hue:   metadata column to colour points by (e.g. Study, Sex, or Age for a colour ramp).
col:   optional column to facet the scatter into side-by-side panels (e.g. Sex).
name:  base label used in the output filename.

Output:
  <output_dir>/pca/<name>_by_<hue>.png
  <output_dir>/pca/<name>_by_<hue>_col_<col>.png

Usage:
  python plot_pca.py \\
      --data_csv   ../../output/qc/dset_qc_v5/data.csv \\
      --var_y DLMUSE_601 DLMUSE_601_corr DLMUSE_509 DLMUSE_509_corr DLMUSE_702 \\
      --hue Study \\
      --name roi_pca \\
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
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PLOT_NAME = "pca"

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font_scale=0.9)

_log = logging.getLogger(__name__)


def _is_numeric_col(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _draw_panel(
    ax,
    pca_df: pd.DataFrame,
    hue: str | None,
    var_explained: list[float],
    title_suffix: str = "",
):
    """Draw one PCA scatter panel on ax."""

    if hue and hue in pca_df.columns:
        col_data = pca_df[hue]
        if _is_numeric_col(col_data):
            sc = ax.scatter(
                pca_df["PC1"], pca_df["PC2"],
                c=col_data, cmap="viridis", s=6, alpha=0.5, linewidths=0,
            )
            plt.colorbar(sc, ax=ax, shrink=0.7, label=hue)
        else:
            groups = sorted(col_data.unique())
            palette = sns.color_palette("tab10", n_colors=len(groups))
            for g, clr in zip(groups, palette):
                mask = col_data == g
                ax.scatter(
                    pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"],
                    color=clr, s=6, alpha=0.5, linewidths=0, label=str(g),
                )
            ax.legend(
                title=hue, fontsize=6, title_fontsize=7,
                markerscale=2, loc="best", framealpha=0.6,
            )
    else:
        ax.scatter(pca_df["PC1"], pca_df["PC2"], s=6, alpha=0.4, linewidths=0)

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)", fontsize=8)
    if title_suffix:
        ax.set_title(title_suffix, fontsize=9)


def plot_pca(
    data_csv: Path,
    var_y: list[str],
    name: str,
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

    missing = [c for c in var_y if c not in df.columns]
    if missing:
        _log.error(f"Columns not found in data: {missing}")
        sys.exit(1)
    if hue and hue not in df.columns:
        _log.error(f"Hue column not found in data: '{hue}'")
        sys.exit(1)
    if col and col not in df.columns:
        _log.error(f"Col column not found in data: '{col}'")
        sys.exit(1)

    if len(var_y) < 2:
        _log.error("plot_pca requires at least 2 columns in var_y.")
        sys.exit(1)

    # keep all auxiliary columns for colouring/faceting; deduplicate to avoid
    # the case where hue and col are the same column (e.g. both "Sex"), which
    # would make df[all_cols]["Sex"] return a 2-D DataFrame instead of a Series.
    aux_cols = list(dict.fromkeys(c for c in ([hue, col] if col else [hue]) if c))
    all_cols = list(dict.fromkeys(var_y + aux_cols))
    data = df[all_cols].dropna(subset=var_y).reset_index(drop=True)

    # standardise and fit PCA on all complete rows
    X = StandardScaler().fit_transform(data[var_y].values)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    var_explained = (pca.explained_variance_ratio_ * 100).tolist()

    pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1]})
    for c in aux_cols:
        pca_df[c] = data[c].values

    if col:
        col_vals = sorted(pca_df[col].dropna().unique())
        n_panels = len(col_vals)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), sharey=True, sharex=True)
        if n_panels == 1:
            axes = [axes]

        for ax, val in zip(axes, col_vals):
            subset = pca_df[pca_df[col] == val]
            _draw_panel(ax, subset, hue, var_explained, title_suffix=f"{col} = {val}")
            if ax is not axes[0]:
                ax.set_ylabel("")

        suptitle = title or (f"PCA of {name}" + (f" — by {hue}" if hue else "") + f" | facet: {col}")
        fig.suptitle(suptitle, y=1.02, fontsize=10)

    else:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        _draw_panel(ax, pca_df, hue, var_explained)
        ax.set_title(title or (f"PCA of {name}" + (f" — by {hue}" if hue else "")))

    n_rows, n_feat = len(pca_df), len(var_y)
    fig.text(
        0.5, -0.02,
        f"n={n_rows} rows · {n_feat} features · PC1+PC2 = {sum(var_explained):.1f}% variance",
        ha="center", fontsize=7, color="dimgrey",
    )

    fig.tight_layout()

    parts = [name]
    if hue:
        parts += [f"by_{hue}"]
    if col:
        parts += [f"col_{col}"]
    out_path = output_dir / plot_name / ("_".join(parts) + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        _log.info(f"Skipping (exists): {out_path}")
        plt.close(fig)
        return

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log.info(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="PCA scatter plot for QC.")
    parser.add_argument("--data_csv",   required=True)
    parser.add_argument("--var_y",      required=True, nargs="+", help="feature columns for PCA")
    parser.add_argument("--name",       required=True, help="base label for output filename")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plot_name",  default=PLOT_NAME)
    parser.add_argument("--hue",        default=None, help="column to colour points by")
    parser.add_argument("--col",        default=None, help="column to facet into side-by-side panels")
    parser.add_argument("--title",   default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--log_dir", default=None)
    args = parser.parse_args()

    setup_logger(__name__, verbose=args.verbose,
                 log_dir=Path(args.log_dir) if args.log_dir else None)

    plot_pca(
        data_csv=Path(args.data_csv),
        var_y=args.var_y,
        name=args.name,
        output_dir=Path(args.output_dir),
        plot_name=args.plot_name,
        hue=args.hue,
        col=args.col,
        title=args.title,
    )


if __name__ == "__main__":
    main()
