"""
Build a full-dataset CSV for one data task (no sample selection).

Reads a flat desc JSON (e.g. data_desc.json) that specifies:
  data_vars  — list of var-group keys from dict_var_groups.json
  filters    — optional dict of column → condition applied after the join:
                 list form:  {"DX_AD": ["CN"]}           keep rows where value in list
                 range form: {"Age": {"min": 50, "max": 90}}
               filter columns not in data_vars are loaded on-demand via var_groups
  add_vars   — optional dict of new_col_name → rule, where rule is either:
                 • a Python expression string evaluated against the full DataFrame
                   (each column available as a pandas Series):
                   {"Study-study2": "Study=='study2'"}          → bool (0/1)
                   {"DLMUSE_601B": "DLMUSE_601/DLMUSE_702*DLMUSE_702.mean()"}
                 • a dict describing a structured transform:
                   {"DLMUSE_601_resid": {"type": "regression_residuals",
                                         "target": "DLMUSE_601",
                                         "covariates": ["Age", "Sex"],
                                         "spline_covariates": {"Age": 5}}}
                   Fits OLS(target ~ covariates) and returns residuals + target.mean()
                   so the corrected values stay on the original scale.
                   spline_covariates: {col: df} replaces the linear term for that
                   covariate with a natural cubic spline of the given degrees of freedom.
  out_dir    — output sub-directory name under output_dir
  out_file   — output CSV filename

Column groups are resolved by looking up each key in dict_var_groups.json (column
list) and finding the corresponding data file via dict_data_files.json (file_prefix).

All MRIDs present in every requested feature file are included (inner join),
then filtered by any conditions in "filters".
Output is written to <output_dir>/<out_dir>/<out_file>.

Usage:
  python merge_all.py \\
      --task_dir   ../../input_anon/tasks/dset_qc_v1 \\
      --desc_file  data_desc.json \\
      --var_groups ../../input_anon/dictionaries/dict_var_groups.json \\
      --data_files ../../input_anon/dictionaries/dict_data_files.json \\
      --data_dir   ../../output/final/data \\
      --output_dir ../../output/intermediate
"""

import argparse
import csv
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

_PIPELINES = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PIPELINES / "utils"))
from logger import setup_logger  # noqa: E402

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "var_groups": THIS_DIR / "../../../input_anon/dictionaries/dict_var_groups.json",
    "data_files": THIS_DIR / "../../../input_anon/dictionaries/dict_data_files.json",
    "data_dir":   THIS_DIR / "../../output/final/data",
    "output_dir": THIS_DIR / "../../output/intermediate",
}

_log = logging.getLogger(__name__)


# ── JSON / CSV helpers ────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if not path.exists():
        _log.error(f"File not found: {path}")
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            _log.error(f"JSON parse error in {path}: {e}")
            sys.exit(1)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def index_by_mrid(rows: list[dict], key: str = "MRID") -> dict[str, dict]:
    return {row[key]: row for row in rows}


# ── var-group resolution ──────────────────────────────────────────────────────

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


def _find_file_prefix(group_key: str, data_files: dict) -> str | None:
    """Return the file_prefix for the data_files entry whose var_groups list contains group_key."""
    for entry in data_files.values():
        if group_key in entry.get("var_groups", []):
            return entry["file_prefix"]
    return None


def _find_file_for_columns(columns: list[str], var_groups: dict, data_files: dict) -> str | None:
    """Return the file_prefix for the first data file whose var_group contains all given columns."""
    non_mrid = [c for c in columns if c != "MRID"]
    for group_key, group_cols in var_groups.items():
        if not group_cols:
            continue
        if all(c in group_cols for c in non_mrid):
            fp = _find_file_prefix(group_key, data_files)
            if fp:
                return fp
    return None


def _load_col_from_data_dir(
    col: str, var_groups: dict, data_files: dict, data_dir: Path
) -> dict[str, str]:
    """Search var_groups for a column; return {MRID: value} or {}."""
    for group_key, columns in var_groups.items():
        if columns and col not in columns:
            continue
        file_prefix = _find_file_prefix(group_key, data_files)
        if not file_prefix:
            continue
        data_path = data_dir / f"{file_prefix}.csv"
        if not data_path.exists():
            continue
        _, rows = read_csv(data_path)
        if rows and col in rows[0]:
            return {r["MRID"]: r[col] for r in rows if "MRID" in r}
    return {}


# ── filtering ─────────────────────────────────────────────────────────────────

def _apply_filters(
    valid_mrids: list[str],
    filters: dict,
    feature_index: dict[str, dict],
    var_groups: dict,
    data_files: dict,
    data_dir: Path,
) -> list[str]:
    remaining = set(valid_mrids)

    for col, condition in filters.items():
        sample = next(iter(feature_index.values()), {})
        if col in sample:
            col_vals = {mrid: feature_index[mrid][col] for mrid in remaining}
        else:
            col_vals = _load_col_from_data_dir(col, var_groups, data_files, data_dir)
            if not col_vals:
                _log.warning(f"filter column '{col}' not found, skipping.")
                continue

        if isinstance(condition, list):
            allowed = {str(v) for v in condition}
            keep = {mrid for mrid in remaining if str(col_vals.get(mrid, "")) in allowed}
        elif isinstance(condition, dict):
            lo, hi = condition.get("min"), condition.get("max")
            keep = set()
            for mrid in remaining:
                try:
                    v = float(col_vals[mrid])
                    if lo is not None and v < lo:
                        continue
                    if hi is not None and v > hi:
                        continue
                    keep.add(mrid)
                except (KeyError, ValueError, TypeError):
                    pass
        else:
            _log.warning(f"unknown filter condition for '{col}', skipping.")
            continue

        n_dropped = len(remaining) - len(keep)
        if n_dropped:
            _log.info(f"  Filter '{col}': {n_dropped} rows dropped → {len(keep)} remaining")
        remaining = keep

    return sorted(remaining)


# ── regression residuals ──────────────────────────────────────────────────────

def _build_design_matrix(sub: pd.DataFrame, covariates: list[str],
                          spline_covariates: dict[str, int]) -> np.ndarray:
    """Build the covariate design matrix.

    covariates with an entry in spline_covariates get a natural cubic spline
    basis (via patsy cr()); the rest are included linearly.  Non-numeric
    columns are one-hot encoded by patsy automatically.
    """
    if spline_covariates:
        from patsy import dmatrix
        parts = []
        for c in covariates:
            if c in spline_covariates:
                df_val = spline_covariates[c]
                parts.append(f"cr(Q('{c}'), df={df_val})")
            else:
                parts.append(f"Q('{c}')")
        formula = " + ".join(parts)
        _log.debug(f"  Design matrix formula: {formula}")
        return np.asarray(dmatrix(formula, sub))
    else:
        sub_cov = sub[covariates].apply(pd.to_numeric, errors="ignore")
        return pd.get_dummies(sub_cov, drop_first=True).astype(float).values


def _regression_residuals(
    df: pd.DataFrame,
    target: str,
    covariates: list[str],
    spline_covariates: dict[str, int] | None = None,
) -> pd.Series:
    """Return target residuals after regressing out covariates, shifted back to target.mean().

    spline_covariates: {col_name: df} — use a natural cubic spline with the
        given degrees of freedom for that covariate instead of a linear term.
    Categorical covariates (e.g. Sex: M/F) are one-hot encoded automatically.
    Rows with any missing value in target or covariates receive NaN in the output.
    """
    spline_covariates = spline_covariates or {}

    cols = [target] + covariates
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _log.error(f"regression_residuals: columns not found in data: {missing}")
        sys.exit(1)

    sub = df.loc[df[cols].notna().all(axis=1)].copy()
    for c in spline_covariates:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    mask = sub[cols].notna().all(axis=1)
    sub  = sub.loc[mask]

    _log.debug(f"  regression_residuals: {len(sub)} complete rows out of {len(df)}")

    X = _build_design_matrix(sub, covariates, spline_covariates)
    y = pd.to_numeric(sub[target], errors="coerce").values.astype(float)

    model     = LinearRegression().fit(X, y)
    residuals = y - model.predict(X) + float(np.mean(y))

    result = pd.Series(np.nan, index=df.index, dtype=float)
    result.loc[sub.index] = residuals
    return result


def _apply_add_var(df: pd.DataFrame, col_name: str, rule) -> pd.Series:
    """Dispatch a single add_vars rule (string expression or dict) to the right handler."""
    if isinstance(rule, str):
        env = {c: df[c] for c in df.columns if c != "MRID"}
        return eval(rule, {}, env)

    if isinstance(rule, dict):
        rtype = rule.get("type")
        if rtype == "regression_residuals":
            return _regression_residuals(
                df, rule["target"], rule["covariates"],
                spline_covariates=rule.get("spline_covariates"),
            )
        _log.error(f"add_vars '{col_name}': unknown rule type '{rtype}'.")
        sys.exit(1)

    _log.error(f"add_vars '{col_name}': rule must be a string or a dict, got {type(rule).__name__}.")
    sys.exit(1)


# ── main function ─────────────────────────────────────────────────────────────

def merge_all(
    task_dir: Path,
    desc_file: str,
    var_groups_path: Path,
    data_files_path: Path,
    data_dir: Path,
    output_dir: Path,
):
    desc_path = task_dir / desc_file
    if not desc_path.exists():
        _log.error(f"Desc file not found: {desc_path}")
        sys.exit(1)

    desc = load_json(desc_path)

    in_vars      = desc["data_vars"]
    filters      = desc.get("filters", {})
    add_vars     = desc.get("add_vars", {})
    out_filename = desc["out_file"]
    out_dir_name = desc.get("out_dir", "")

    var_groups = _resolve_var_groups(load_json(var_groups_path))
    data_files = load_json(data_files_path)

    _log.info(f"Desc    : {desc_path.name}")

    # ── load feature data ─────────────────────────────────────────────────────
    feature_cols: list[str] = []
    feature_index: dict[str, dict] = {}
    group_mrid_sets: list[set[str]] = []

    for item in in_vars:
        if isinstance(item, list):
            columns     = item
            display_key = f"[{', '.join(str(c) for c in item)}]"
            file_prefix = _find_file_for_columns(columns, var_groups, data_files)
            if file_prefix is None:
                _log.warning(f"no data file found for inline columns {item}, skipping.")
                continue
        else:
            key = item
            if key not in var_groups:
                _log.warning(f"'{key}' not in var_groups dict, skipping.")
                continue
            columns     = var_groups[key]
            display_key = key
            file_prefix = _find_file_prefix(key, data_files)
            if file_prefix is None:
                _log.warning(f"no data_files entry contains var_group '{key}', skipping.")
                continue

        data_path = data_dir / f"{file_prefix}.csv"
        if not data_path.exists():
            _log.warning(f"data file not found ({file_prefix}.csv), skipping.")
            continue

        fieldnames, data_rows = read_csv(data_path)
        data_by_mrid = index_by_mrid(data_rows)

        group_cols = [c for c in columns if c != "MRID"]
        if not group_cols:
            cols = [c for c in fieldnames if c != "MRID"]
        else:
            missing = [c for c in group_cols if c not in fieldnames]
            if missing:
                _log.warning(f"columns not in {file_prefix}.csv for {display_key} (skipped): {missing}")
            cols = [c for c in group_cols if c in fieldnames]

        for mrid, feat_row in data_by_mrid.items():
            feature_index.setdefault(mrid, {}).update({c: feat_row[c] for c in cols})

        feature_cols.extend(cols)
        group_mrid_sets.append(set(data_by_mrid.keys()))
        _log.info(f"  {display_key} ({file_prefix}.csv)  — {len(cols)} columns, {len(data_by_mrid)} MRIDs")

    if not group_mrid_sets:
        _log.error("No feature data loaded. Aborting.")
        sys.exit(1)

    valid_mrids = sorted(set.intersection(*group_mrid_sets))
    n_dropped = len(feature_index) - len(valid_mrids)
    if n_dropped:
        _log.info(f"  ({n_dropped} MRIDs dropped — absent in one or more feature files)")

    if filters:
        _log.info(f"Filters : {list(filters.keys())}")
        valid_mrids = _apply_filters(
            valid_mrids, filters, feature_index, var_groups, data_files, data_dir
        )

    # ── output ────────────────────────────────────────────────────────────────
    actual_out_dir = (output_dir / out_dir_name) if out_dir_name else output_dir
    actual_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = actual_out_dir / out_filename

    if out_path.exists():
        _log.info(f"Skipping: {out_path} already exists.")
        return

    if add_vars:
        _log.info(f"add_vars: {list(add_vars.keys())}")

    out_df = pd.DataFrame(
        [{**{"MRID": mrid}, **feature_index[mrid]} for mrid in valid_mrids],
        columns=["MRID"] + feature_cols,
    )
    for col in feature_cols:
        try:
            out_df[col] = pd.to_numeric(out_df[col])
        except (ValueError, TypeError):
            pass

    for col_name, rule in add_vars.items():
        _log.debug(f"  add_vars: computing '{col_name}'")
        try:
            out_df[col_name] = _apply_add_var(out_df, col_name, rule)
        except Exception as e:
            _log.error(f"add_vars '{col_name}' failed — {e}")
            _log.debug(traceback.format_exc())
            out_df[col_name] = float("nan")

    out_df.to_csv(out_path, index=False)

    n_derived = len(add_vars)
    _log.info(f"Output  : {out_path}  ({len(out_df)} rows, {len(feature_cols)} features"
              + (f", {n_derived} derived" if n_derived else "") + ")")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build a full-dataset CSV from a flat task desc JSON (no sample selection)."
    )
    parser.add_argument("--task_dir",   required=True,
                        help="path to task directory containing desc JSON")
    parser.add_argument("--desc_file",  required=True,
                        help="desc JSON filename (relative to task_dir), e.g. data_desc.json")
    parser.add_argument("--var_groups", default=DEFAULTS["var_groups"],
                        help="path to dict_var_groups.json")
    parser.add_argument("--data_files", default=DEFAULTS["data_files"],
                        help="path to dict_data_files.json")
    parser.add_argument("--data_dir",   default=DEFAULTS["data_dir"],
                        help="directory containing input feature CSVs")
    parser.add_argument("--output_dir", default=DEFAULTS["output_dir"],
                        help="base output directory; output goes to <output_dir>/<out_dir>/<out_file>")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="show DEBUG messages on console")
    parser.add_argument("--log_dir",  default=None,
                        help="directory for WARNING/ERROR log file (default: no log file)")
    args = parser.parse_args()

    setup_logger(
        __name__,
        verbose=args.verbose,
        log_dir=Path(args.log_dir) if args.log_dir else None,
    )

    merge_all(
        task_dir=Path(args.task_dir),
        desc_file=args.desc_file,
        var_groups_path=Path(args.var_groups),
        data_files_path=Path(args.data_files),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
