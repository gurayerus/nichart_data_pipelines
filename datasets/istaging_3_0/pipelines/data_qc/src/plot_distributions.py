"""
Generate distribution plots for a verification task.

Reads plot specifications from <verifications_dir>/<name>/verif_desc.json and
produces one PNG per plot entry.

Supported plot types:
  hist_by_group  — KDE density curves, one per group value
  boxplot        — box plot (x = categorical, y = continuous)
  scatter        — scatter plot with optional color grouping

Usage:
  python plot_distributions.py dlmuse601_distributions \
      --data_csv /path/to/data.csv --output_dir /path/to/plots
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "verifications_dir": THIS_DIR / "../../../input_anon/verifications",
}

sns.set_theme(style="whitegrid", font_scale=0.9)


def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"File not found: {path}")
    with path.open(encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(f"JSON parse error in {path}: {e}")


def _fig_width_for_categories(n: int, min_w: float = 8.0, per_cat: float = 0.35) -> float:
    return max(min_w, n * per_cat)


def plot_hist_by_group(df: pd.DataFrame, cfg: dict, out_path: Path):
    var      = cfg["variable"]
    group_by = cfg["group_by"]
    title    = cfg.get("title", f"{var} by {group_by}")

    data     = df[[var, group_by]].dropna()
    groups   = sorted(data[group_by].unique())
    n_groups = len(groups)

    palette = sns.color_palette("husl", n_colors=n_groups)
    fig, ax = plt.subplots(figsize=(10, 5))

    for grp, color in zip(groups, palette):
        subset = data.loc[data[group_by] == grp, var]
        if len(subset) < 2:
            continue
        try:
            sns.kdeplot(subset, ax=ax, label=str(grp), color=color, alpha=0.65, linewidth=1.2)
        except Exception:
            pass

    ax.set_xlabel(var)
    ax.set_ylabel("Density")
    ax.set_title(title)

    if n_groups <= 20:
        ax.legend(title=group_by, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    else:
        ax.legend_.remove() if ax.get_legend() else None
        ax.set_title(f"{title}  ({n_groups} groups — legend omitted)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_boxplot(df: pd.DataFrame, cfg: dict, out_path: Path):
    x     = cfg["x"]
    y     = cfg["y"]
    title = cfg.get("title", f"{y} by {x}")

    data    = df[[x, y]].dropna()
    order   = sorted(data[x].unique())
    n_cats  = len(order)
    fig_w   = _fig_width_for_categories(n_cats)

    fig, ax = plt.subplots(figsize=(fig_w, 5))
    sns.boxplot(data=data, x=x, y=y, order=order, ax=ax,
                linewidth=0.7, flierprops={"markersize": 1.5, "alpha": 0.4})

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if n_cats > 8:
        ax.tick_params(axis="x", labelrotation=90, labelsize=max(4, 8 - n_cats // 10))

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_scatter(df: pd.DataFrame, cfg: dict, out_path: Path):
    x        = cfg["x"]
    y        = cfg["y"]
    color_by = cfg.get("color_by")
    title    = cfg.get("title", f"{y} vs {x}")

    cols = [c for c in [x, y, color_by] if c]
    data = df[cols].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    if color_by:
        groups  = sorted(data[color_by].unique())
        palette = sns.color_palette("Set1", n_colors=len(groups))
        for grp, color in zip(groups, palette):
            sub = data[data[color_by] == grp]
            ax.scatter(sub[x], sub[y], s=4, alpha=0.35, color=color, label=str(grp))
        ax.legend(title=color_by)
    else:
        ax.scatter(data[x], data[y], s=4, alpha=0.3)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


_PLOT_FNS = {
    "hist_by_group": plot_hist_by_group,
    "boxplot":       plot_boxplot,
    "scatter":       plot_scatter,
}


def plot_distributions(
    verif_name: str,
    verifications_dir: Path,
    data_csv: Path,
    output_dir: Path,
):
    verif_dir  = verifications_dir / verif_name
    desc       = load_json(verif_dir / "verif_desc.json")
    plot_specs = desc.get("plots", [])

    if not plot_specs:
        print("No plots defined in verif_desc.json.")
        return

    if not data_csv.exists():
        sys.exit(f"Data CSV not found: {data_csv}")

    df = pd.read_csv(data_csv, low_memory=False)
    print(f"Loaded: {data_csv.name}  ({len(df)} rows, {len(df.columns)} columns)")

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, spec in enumerate(plot_specs):
        ptype = spec.get("type")
        fn    = _PLOT_FNS.get(ptype)
        if fn is None:
            print(f"Warning: unknown plot type '{ptype}', skipping.", file=sys.stderr)
            continue

        out_file = spec.get("file") or f"plot_{i:02d}_{ptype}.png"
        out_path = output_dir / out_file

        if out_path.exists():
            print(f"  Skipping (exists): {out_path.name}")
            continue

        try:
            fn(df, spec, out_path)
        except Exception as e:
            print(f"Warning: plot '{out_file}' failed: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate distribution plots for a verification task.")
    parser.add_argument("verif_name",
                        help="verification name (must match a subdirectory in verifications_dir)")
    parser.add_argument("--verifications_dir", default=DEFAULTS["verifications_dir"],
                        help="directory containing verification subdirectories")
    parser.add_argument("--data_csv",   required=True, help="path to merged data CSV")
    parser.add_argument("--output_dir", required=True, help="directory for output plot PNGs")
    args = parser.parse_args()

    plot_distributions(
        verif_name=args.verif_name,
        verifications_dir=Path(args.verifications_dir),
        data_csv=Path(args.data_csv),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
