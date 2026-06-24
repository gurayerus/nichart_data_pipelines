"""
Split the iSTAGING 3.0 input CSV into themed sub-tables.

Column assignments are resolved from two JSON files:

  dict_data_files.json  — maps file-key → {type, file_prefix, var_groups: [...]}
  dict_var_groups.json  — maps group-key → [col1, col2, ...]

Only entries with type == "init" are processed (they come from the raw input CSV).
The column list for each output file is taken from var_groups[file_key] (the
canonical group that shares the file key's name).

Use -g to write only a subset of groups (space-separated dict_data_files keys).

Usage:
  python split_input.py                          # all init groups in the dict
  python split_input.py -g visits demog
  python split_input.py -i <csv> -d <dict_data_files> -v <dict_var_groups> -o <outdir>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "input_csv":  THIS_DIR / "../../../input/data_anon/istaging_3_0_test_v4_anon.csv",
    "var_dict":   THIS_DIR / "../../../input_anon/dictionaries/dict_data_files.json",
    "var_groups": THIS_DIR / "../../../input_anon/dictionaries/dict_var_groups.json",
    "output_dir": THIS_DIR / "../output",
}


def _resolve_var_groups(var_groups: dict) -> dict:
    """Expand derived group definitions into plain column lists.

    Supported forms:
      {"concat": ["group_a", "group_b", ...]}   — concatenate other groups
      {"prefix": "H_", "from": "group"}         — prefix every column from another group
    """
    resolved = {}
    for key, value in var_groups.items():
        if isinstance(value, dict) and "concat" in value:
            cols = []
            for g in value["concat"]:
                cols.extend(resolved.get(g, []))
            resolved[key] = cols
        elif isinstance(value, dict) and "prefix" in value and "from" in value:
            base = resolved.get(value["from"], [])
            resolved[key] = [value["prefix"] + c for c in base]
        else:
            resolved[key] = value
    return resolved


def load_dict(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(f"JSON parse error in {path}: {e}")


def split(
    input_csv: Path,
    var_dict: Path,
    var_groups_path: Path,
    output_dir: Path,
    groups: list[str] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = load_dict(var_dict)
    var_groups  = _resolve_var_groups(load_dict(var_groups_path))

    if groups is not None:
        unknown = set(groups) - set(data_files)
        if unknown:
            sys.exit(f"Unknown group(s): {sorted(unknown)}. Available: {sorted(data_files)}")
        entries = {k: v for k, v in data_files.items() if k in groups}
    else:
        # default: only process init-type files
        entries = {k: v for k, v in data_files.items() if v.get("type") == "init"}

    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_cols = set(reader.fieldnames)
        rows = list(reader)

    print(f"Input : {input_csv}  ({len(rows)} rows, {len(csv_cols)} columns)")
    print(f"Output: {output_dir}")

    for key, entry in entries.items():
        file_prefix = entry["file_prefix"]

        # resolve column list from the canonical var_group (same name as the file key)
        if key not in var_groups:
            print(f"  Warning: skipping '{key}' — no var_group named '{key}' found.",
                  file=sys.stderr)
            continue
        columns = var_groups[key]
        if not columns:
            print(f"  Warning: skipping '{key}' — var_group '{key}' is empty.",
                  file=sys.stderr)
            continue

        present = [c for c in columns if c in csv_cols]
        missing = [c for c in columns if c not in csv_cols]

        # MRID is always the key column — prepend it if present in the input
        if "MRID" in csv_cols and "MRID" not in present:
            present = ["MRID"] + present

        stem = f"{file_prefix}_missingvars" if missing else file_prefix
        out_path = output_dir / f"{stem}.csv"

        if out_path.exists():
            print(f"  Warning: skipping '{key}' — output file already exists: {out_path}",
                  file=sys.stderr)
            continue

        data_cols = [c for c in present if c != "MRID"]
        if not data_cols:
            print(f"  Warning: skipping '{key}' — no data columns available: {missing}",
                  file=sys.stderr)
            continue

        if missing:
            print(f"  Warning: '{key}' — {len(missing)} column(s) missing, "
                  f"writing as {stem}.csv: {missing}", file=sys.stderr)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=present, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"  {stem}.csv  — {len(present)}/{len(columns)} columns")


def main():
    parser = argparse.ArgumentParser(description="Split iSTAGING CSV into themed sub-tables.")
    parser.add_argument("-i", "--input_csv",  default=DEFAULTS["input_csv"],
                        help="path to input CSV")
    parser.add_argument("-d", "--var_dict",   default=DEFAULTS["var_dict"],
                        help="path to dict_data_files.json")
    parser.add_argument("-v", "--var_groups", default=DEFAULTS["var_groups"],
                        help="path to dict_var_groups.json")
    parser.add_argument("-o", "--output_dir", default=DEFAULTS["output_dir"],
                        help="directory for output CSVs")
    parser.add_argument("-g", "--groups", nargs="+", default=None, metavar="KEY",
                        help="data_files keys to write (default: all init entries)")
    args = parser.parse_args()

    split(
        input_csv=Path(args.input_csv),
        var_dict=Path(args.var_dict),
        var_groups_path=Path(args.var_groups),
        output_dir=Path(args.output_dir),
        groups=args.groups,
    )


if __name__ == "__main__":
    main()
