"""
Build a merged feature CSV from a task desc JSON (new format only).

Desc JSON keys required: sample, data, output.

  sample:
    path:    sample directory (relative to --root)
    file:    sample list CSV filename
    columns: columns to keep from the sample list (MRID + extras like 'batch')
    (empty dict {} -> use all MRIDs from the merged feature data)
  data:
    path:            data directory (relative to --root)
    columns:         var-group keys or inline column lists
    filters:         per-column filter (optional):
                       {col: {min: X, max: Y}}  -- numeric range
                       {col: ["v1", "v2"]}       -- include-list (keep matching rows)
    additional_vars: {new_col: "expr"}           -- pandas eval expression (supports .mean() etc.) (optional)
                     {new_col: {"type": "regression_residuals", "target": "col",
                                "covariates": [...], "spline_covariates": {col: n_knots}}}
    mappings:        {col: {from: to}}           -- value remapping (optional)
  output:
    path: output directory (relative to --root)
    file: output CSV filename

Usage:
  python merge_files.py \\
      --task_dir  input_anon/tasks/dset_qc_v2 \\
      --desc_file data_desc.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "root":       THIS_DIR / "../../../..",
    "var_groups": THIS_DIR / "../../../input_anon/dictionaries/dict_var_groups.json",
    "data_files": THIS_DIR / "../../../input_anon/dictionaries/dict_data_files.json",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"File not found: {path}")
    with path.open(encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            sys.exit(f"JSON parse error in {path}: {e}")


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def index_by_mrid(rows: list[dict], key: str = "MRID") -> dict[str, dict]:
    return {row[key]: row for row in rows}


def _resolve_var_groups(var_groups: dict) -> dict:
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


def _parse_col_spec(item: str) -> tuple[str, list[str] | str] | None:
    """Parse 'file_key.group_name' or 'file_key.[Col1, Col2]'.

    Returns (file_key, columns) where columns is a list for inline specs
    or a string group name for group specs.  Returns None on bad format.
    """
    if "." not in item:
        return None
    file_key, rest = item.split(".", 1)
    file_key = file_key.strip()
    rest = rest.strip()
    if rest.startswith("["):
        cols = [c.strip() for c in rest.strip("[]").split(",") if c.strip()]
        return file_key, cols
    return file_key, rest


def _load_feature_data(
    column_specs: list,
    data_dir: Path,
    var_groups: dict,
    data_files: dict,
) -> tuple[list[str], dict[str, dict]]:
    feature_cols: list[str] = []
    feature_index: dict[str, dict] = {}

    for item in column_specs:
        parsed = _parse_col_spec(item)
        if parsed is None:
            print(f"Warning: '{item}' — expected 'file_key.group' or "
                  f"'file_key.[Col1, Col2]', skipping.", file=sys.stderr)
            continue

        file_key, group_or_cols = parsed

        if file_key not in data_files:
            print(f"Warning: '{file_key}' not in dict_data_files.json, skipping.",
                  file=sys.stderr)
            continue
        file_prefix = data_files[file_key]["file_prefix"]

        if isinstance(group_or_cols, list):
            columns = group_or_cols
        else:
            group_name = group_or_cols
            if group_name not in var_groups:
                print(f"Warning: '{group_name}' not in var_groups dict, skipping.",
                      file=sys.stderr)
                continue
            columns = var_groups[group_name]

        data_path = data_dir / f"{file_prefix}.csv"
        if not data_path.exists():
            print(f"Warning: data file not found ({file_prefix}.csv), skipping.",
                  file=sys.stderr)
            continue

        fieldnames, data_rows = read_csv(data_path)
        data_by_mrid = index_by_mrid(data_rows)

        group_cols = [c for c in columns if c != "MRID"]
        if not group_cols:
            cols = [c for c in fieldnames if c != "MRID"]
        else:
            missing = [c for c in group_cols if c not in fieldnames]
            if missing:
                print(f"Warning: columns not in {file_prefix}.csv for '{item}' "
                      f"(skipped): {missing}", file=sys.stderr)
            cols = [c for c in group_cols if c in fieldnames]

        for mrid, feat_row in data_by_mrid.items():
            feature_index.setdefault(mrid, {}).update({c: feat_row[c] for c in cols})

        feature_cols.extend(cols)
        print(f"  {item} ({file_prefix}.csv)  -- {len(cols)} columns")

    return feature_cols, feature_index


def _regression_residuals(df, spec: dict):
    """Fit linear (+ optional spline) regression and return residuals."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import SplineTransformer

    target      = spec["target"]
    covariates  = spec.get("covariates", [])
    spline_covs = spec.get("spline_covariates", {})

    cols_needed = [target] + covariates
    valid       = df[cols_needed].dropna()

    if len(valid) == 0:
        return pd.Series(float("nan"), index=df.index)

    y = valid[target].values.astype(float)

    X_parts = []
    for cov in covariates:
        if cov not in spline_covs:
            X_parts.append(valid[[cov]].values.astype(float))
    for cov, n_knots in spline_covs.items():
        st = SplineTransformer(n_knots=int(n_knots), degree=3, include_bias=False)
        X_parts.append(st.fit_transform(valid[[cov]].values.astype(float)))



    if not X_parts:
        return pd.Series(float("nan"), index=df.index)

    X = np.hstack(X_parts)
    residuals = y - LinearRegression().fit(X, y).predict(X)

    result = pd.Series(float("nan"), index=df.index, dtype=float)
    result.loc[valid.index] = residuals

    return result


def _apply_additional_vars(feature_index: dict, additional_vars: dict) -> list[str]:
    if not additional_vars:
        return []

    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas required for additional_vars — skipping.", file=sys.stderr)
        return []

    # Build working DataFrame; coerce numeric columns where possible
    df = pd.DataFrame.from_dict(feature_index, orient="index")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    new_cols: list[str] = []
    for new_col, spec in additional_vars.items():
        try:
            if isinstance(spec, str):
                # DataFrame-level eval — supports column aggregates like .mean()
                df[new_col] = df.eval(spec)
            elif isinstance(spec, dict):
                spec_type = spec.get("type")
                if spec_type == "regression_residuals":
                    df[new_col] = _regression_residuals(df, spec)
                    print(df)
                else:
                    print(f"Warning: unknown additional_var type '{spec_type}' "
                          f"for '{new_col}', skipping.", file=sys.stderr)
                    continue
            else:
                print(f"Warning: unsupported spec for additional_var '{new_col}', skipping.",
                      file=sys.stderr)
                continue

            new_cols.append(new_col)
            n = int(df[new_col].notna().sum())
            print(f"  Additional var '{new_col}'  -- computed for {n} MRIDs")

        except Exception as e:
            print(f"Warning: additional_var '{new_col}' failed: {e}", file=sys.stderr)

    # Write computed columns back to feature_index
    for mrid in feature_index:
        for col in new_cols:
            val = df.at[mrid, col] if mrid in df.index else None
            feature_index[mrid][col] = "" if pd.isna(val) else str(val)

    return new_cols


def _apply_filters(feature_index: dict, filters: dict) -> dict:
    for col, filt in filters.items():
        n_before = len(feature_index)
        keep = {}
        for mrid, row in feature_index.items():
            val = row.get(col)
            if val is None or val == "":
                continue
            if isinstance(filt, list):
                # include-list: keep rows whose value is in the list
                if str(val) in [str(v) for v in filt]:
                    keep[mrid] = row
            else:
                # numeric range: {min: X, max: Y}
                try:
                    v = float(val)
                except ValueError:
                    continue
                if "min" in filt and v < filt["min"]:
                    continue
                if "max" in filt and v > filt["max"]:
                    continue
                keep[mrid] = row
        feature_index = keep
        lo = filt if isinstance(filt, list) else f"{filt.get('min', '')}–{filt.get('max', '')}"
        print(f"  Filter '{col}' {lo}: {n_before} -> {len(feature_index)} MRIDs")
    return feature_index




def _apply_mappings(mappings: dict, rows_list: list[dict]) -> None:
    for col, col_map in mappings.items():
        str_map = {str(k): v for k, v in col_map.items()}
        unmapped = {str(r[col]) for r in rows_list if col in r and str(r[col]) not in str_map}
        if unmapped:
            print(f"Warning: '{col}' values not in mappings (kept as-is): {sorted(unmapped)}",
                  file=sys.stderr)
        for row in rows_list:
            if col in row and str(row[col]) in str_map:
                row[col] = str(str_map[str(row[col])])


def _write_output(out_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Skipping: {out_path} already exists.")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── main logic ─────────────────────────────────────────────────────────────────

def merge_files(
    task_dir: Path,
    desc_file: str,
    var_groups_path: Path,
    data_files_path: Path,
    root: Path,
    output_root: Path | None = None,
) -> None:
    desc_path = task_dir / desc_file
    if not desc_path.exists():
        sys.exit(f"Desc file not found: {desc_path}")

    desc = load_json(desc_path)
    if not all(k in desc for k in ("sample", "data", "output")):
        sys.exit(f"Desc file must have 'sample', 'data', and 'output' keys: {desc_path}")

    print(f"Desc    : {desc_path.name}")

    sample_sec = desc.get("sample", {})
    data_sec   = desc["data"]
    out_sec    = desc["output"]

    oroot    = output_root if output_root is not None else root
    data_dir = oroot / data_sec["path"]
    out_path = oroot / out_sec["path"] / out_sec["file"]

    var_groups = _resolve_var_groups(load_json(var_groups_path))
    data_files = load_json(data_files_path)

    # ── feature data ──────────────────────────────────────────────────────────
    feature_cols, feature_index = _load_feature_data(
        data_sec["columns"], data_dir, var_groups, data_files
    )

    # ── sample list ───────────────────────────────────────────────────────────
    if sample_sec:
        sample_path = root / sample_sec["path"] / sample_sec["file"]
        if not sample_path.exists():
            sys.exit(f"Sample list not found: {sample_path}")
        _, sample_rows = read_csv(sample_path)
        sample_index = index_by_mrid(sample_rows)
        sample_extra = [c for c in sample_sec.get("columns", []) if c != "MRID"]
        print(f"List    : {sample_path.name}  ({len(sample_index)} MRIDs)")
    else:
        sample_index = {mrid: {} for mrid in feature_index}
        sample_extra = []
        print(f"List    : all feature MRIDs ({len(sample_index)} MRIDs)")

    # ── merge ─────────────────────────────────────────────────────────────────
    out_fieldnames = ["MRID"] + sample_extra + feature_cols
    merged_index: dict[str, dict] = {}
    n_missing = 0

    for mrid, sample_row in sample_index.items():
        if mrid not in feature_index:
            n_missing += 1
            continue
        row = {"MRID": mrid}
        row.update({c: sample_row.get(c, "") for c in sample_extra})
        row.update(feature_index[mrid])
        merged_index[mrid] = row

    if n_missing:
        print(f"Warning: {n_missing} MRIDs from sample have no feature data (excluded).",
              file=sys.stderr)

    # ── filters ───────────────────────────────────────────────────────────────
    if data_sec.get("filters"):
        merged_index = _apply_filters(merged_index, data_sec["filters"])

    # ── mappings ──────────────────────────────────────────────────────────────
    if data_sec.get("mappings"):
        _apply_mappings(data_sec["mappings"], list(merged_index.values()))

    # ── additional vars ───────────────────────────────────────────────────────
    if data_sec.get("additional_vars"):
        feature_cols += _apply_additional_vars(merged_index, data_sec["additional_vars"])
        out_fieldnames = ["MRID"] + sample_extra + feature_cols

    _write_output(out_path, out_fieldnames, list(merged_index.values()))
    print(f"Output  : {out_path}  ({len(merged_index)} rows, {len(feature_cols)} features)")


def main():
    parser = argparse.ArgumentParser(
        description="Build a merged feature CSV from a task desc JSON."
    )
    parser.add_argument("--task_dir",   required=True,
                        help="path to task directory containing desc JSON")
    parser.add_argument("--desc_file",  required=True,
                        help="desc JSON filename (relative to task_dir)")
    parser.add_argument("--var_groups", default=DEFAULTS["var_groups"],
                        help="path to dict_var_groups.json")
    parser.add_argument("--data_files", default=DEFAULTS["data_files"],
                        help="path to dict_data_files.json")
    parser.add_argument("--root",        default=DEFAULTS["root"],
                        help="root for resolving sample paths (default: 4 levels up from script)")
    parser.add_argument("--output_root", default=None,
                        help="root for resolving data.path and output.path "
                             "(defaults to --root when omitted)")
    args = parser.parse_args()

    merge_files(
        task_dir=Path(args.task_dir),
        desc_file=args.desc_file,
        var_groups_path=Path(args.var_groups),
        data_files_path=Path(args.data_files),
        root=Path(args.root),
        output_root=Path(args.output_root) if args.output_root else None,
    )


if __name__ == "__main__":
    main()
