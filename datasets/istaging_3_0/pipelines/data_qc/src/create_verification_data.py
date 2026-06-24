"""
Build a merged CSV for a verification task.

Reads from <verifications_dir>/<name>/verif_desc.json, loads each variable group
listed under in_data.in_vars from the final data directory (using dict_var_groups.json
to locate the right file and columns), inner-joins on MRID, and writes one output CSV.

Usage:
  python create_verification_data.py dlmuse601_distributions
  python create_verification_data.py dlmuse601_distributions --output_dir /path/to/dir
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "verifications_dir": THIS_DIR / "../../../input_anon/verifications",
    "var_groups":        THIS_DIR / "../../../input_anon/dictionaries/dict_var_groups.json",
    "data_dir":          THIS_DIR / "../../../output/final/data",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"File not found: {path}")
    with path.open(encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(f"JSON parse error in {path}: {e}")


def create_verification_data(
    verif_name: str,
    verifications_dir: Path,
    var_groups_path: Path,
    data_dir: Path,
    output_dir: Path,
):
    verif_dir = verifications_dir / verif_name
    if not verif_dir.exists():
        sys.exit(f"Verification directory not found: {verif_dir}")

    desc       = load_json(verif_dir / "verif_desc.json")
    var_groups = load_json(var_groups_path)

    in_vars  = desc["in_data"]["in_vars"]
    out_file = desc["out_data"]["file"]
    out_path = output_dir / out_file

    if out_path.exists():
        print(f"Skipping: {out_path} already exists.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    merged: pd.DataFrame | None = None

    for key in in_vars:
        if key not in var_groups:
            print(f"Warning: '{key}' not in var_groups, skipping.", file=sys.stderr)
            continue

        group     = var_groups[key]
        file_path = data_dir / f"{group['file_prefix']}.csv"

        if not file_path.exists():
            print(f"Warning: {file_path.name} not found for '{key}', skipping.", file=sys.stderr)
            continue

        df       = pd.read_csv(file_path, low_memory=False)
        req_cols = [c for c in group["columns"] if c != "MRID"] if group["columns"] else [c for c in df.columns if c != "MRID"]
        missing  = [c for c in req_cols if c not in df.columns]
        if missing:
            print(f"Warning: columns absent in {group['file_prefix']}.csv for '{key}' (skipped): {missing}", file=sys.stderr)
        keep_cols = ["MRID"] + [c for c in req_cols if c in df.columns]
        df        = df[keep_cols]

        print(f"  {key} ({group['file_prefix']}.csv) — {len(keep_cols) - 1} columns, {len(df)} rows")

        merged = df if merged is None else merged.merge(df, on="MRID", how="inner")

    if merged is None or merged.empty:
        sys.exit("No data loaded — check var_groups keys and data_dir.")

    merged.to_csv(out_path, index=False)
    n_cols = len(merged.columns) - 1
    print(f"Output: {out_path}  ({len(merged)} rows, {n_cols} feature columns)")


def main():
    parser = argparse.ArgumentParser(description="Build merged CSV for a verification task.")
    parser.add_argument("verif_name",
                        help="verification name (must match a subdirectory in verifications_dir)")
    parser.add_argument("--verifications_dir", default=DEFAULTS["verifications_dir"],
                        help="directory containing verification subdirectories")
    parser.add_argument("--var_groups",        default=DEFAULTS["var_groups"],
                        help="path to dict_var_groups.json")
    parser.add_argument("--data_dir",          default=DEFAULTS["data_dir"],
                        help="directory containing final data CSVs")
    parser.add_argument("--output_dir",        default=None,
                        help="output directory (default: <verif_dir>/output/data)")
    args = parser.parse_args()

    verif_dir  = Path(args.verifications_dir) / args.verif_name
    output_dir = Path(args.output_dir) if args.output_dir else verif_dir / "output" / "data"

    create_verification_data(
        verif_name=args.verif_name,
        verifications_dir=Path(args.verifications_dir),
        var_groups_path=Path(args.var_groups),
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
