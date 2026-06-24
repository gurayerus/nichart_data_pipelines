"""
Univariate batch harmonization.

Input CSV column layout (1-indexed):
  col 1  : MRID (key)
  col 2  : batch variable
  col 3-5: covariates — col 3 (first) gets a natural cubic spline for age
  col 6+ : data variables to harmonize

For each data variable, fits:
  y ~ C(batch) + cr(cov1, df=SPLINE_DF) + cov2 + cov3

Batch effect is removed:
  y_harm = y - X_batch @ beta_batch

Outputs:
  <prefix>.csv      — MRID + harmonized columns (renamed with --col_prefix)
  <prefix>.joblib   — estimated parameters for each variable

Usage:
  python harm_univar.py input.csv
  python harm_univar.py input.csv -o ../output/harmonization -p myrun -c H_
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.regression.linear_model import OLS

THIS_DIR = Path(__file__).parent
SPLINE_DF = 5  # degrees of freedom for natural cubic spline on first covariate (age)

DEFAULTS = {
    "output_dir": THIS_DIR / "../../output/harmonization",
    "out_prefix":  None,   # falls back to input stem
    "col_prefix":  "H2_",  # prefix for harmonized column names in output CSV
}


def harm_univar(input_csv: Path, output_dir: Path, out_prefix: str = "", col_prefix: str = ""):
    prefix    = out_prefix or input_csv.stem
    out_csv   = output_dir / f"{prefix}.csv"
    out_model = output_dir / f"{prefix}.joblib"

    if out_csv.exists() and out_model.exists():
        print(f"Skipping: output files already exist ({out_csv.name}, {out_model.name}).")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        sys.exit(f"Error reading input CSV '{input_csv}': {e}")

    all_cols = df.columns.tolist()

    if len(all_cols) < 6:
        sys.exit(f"Need at least 6 columns (MRID, batch, 3 covariates, 1+ data vars), got {len(all_cols)}.")

    mrid_col   = all_cols[0]
    batch_col  = all_cols[1]
    covar_cols = all_cols[2:5]   # [age_col, cov2, cov3]
    data_cols  = all_cols[5:]

    age_col = covar_cols[0]

    # Map M/F to 1/0 in covariate columns before numeric conversion
    for c in covar_cols:
        df[c] = df[c].replace({"M": 1, "F": 0})

    # Convert covariate and data columns to numeric
    for c in covar_cols + data_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=covar_cols + data_cols).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Warning: dropped {n_before - len(df)} rows with missing values.", file=sys.stderr)

    print(f"Input : {input_csv.name}  ({len(df)} rows, {len(data_cols)} data vars)")
    print(f"  batch col  : {batch_col}  ({df[batch_col].nunique()} levels: {sorted(df[batch_col].unique())})")
    print(f"  covariates : {covar_cols}  (spline on '{age_col}', df={SPLINE_DF})")

    # ── rename to safe names so patsy has no issues with special chars ────────
    safe = {c: f"_v{i}" for i, c in enumerate(all_cols)}
    df_safe = df.rename(columns=safe)

    s_batch  = safe[batch_col]
    s_age    = safe[age_col]
    s_covars = [safe[c] for c in covar_cols[1:]]
    s_data   = [safe[c] for c in data_cols]

    cov_terms = [f"cr({s_age}, df={SPLINE_DF})"] + s_covars
    formula   = f"C({s_batch}) + " + " + ".join(cov_terms)

    # Build design matrix once (same for all variables)
    X_dm = dmatrix(formula, df_safe)
    X    = np.asarray(X_dm)

    # Locate batch term columns via design_info
    batch_slice = None
    for term, slc in X_dm.design_info.term_slices.items():
        if s_batch in str(term):
            batch_slice = slc
            break

    if batch_slice is None:
        sys.exit(f"Could not locate batch term '{batch_col}' in design matrix.")

    X_batch = X[:, batch_slice]
    batch_col_names = X_dm.design_info.column_names[batch_slice]

    # ── fit + harmonize each data variable ────────────────────────────────────
    harm_data  = {}
    model_vars = {}

    for var, s_var in zip(data_cols, s_data):
        y = df_safe[s_var].values.astype(float)
        fit = OLS(y, X).fit()

        beta_batch   = fit.params[batch_slice]
        batch_effect = X_batch @ beta_batch
        y_harm       = y - batch_effect

        out_col = f"{col_prefix}{var}" if col_prefix else var
        harm_data[out_col] = y_harm
        model_vars[var] = {
            "params":      dict(zip(X_dm.design_info.column_names, fit.params)),
            "batch_terms": dict(zip(batch_col_names, beta_batch)),
            "rsquared":    fit.rsquared,
            "nobs":        int(fit.nobs),
        }

    # ── write harmonized CSV (MRID + harmonized columns only) ─────────────────
    out_df = pd.concat(
        [df[[mrid_col]], pd.DataFrame(harm_data, index=df.index)],
        axis=1,
    )
    out_df.to_csv(out_csv, index=False)

    # ── save model ────────────────────────────────────────────────────────────
    model_meta = {
        "input_csv":       str(input_csv),
        "batch_col":       batch_col,
        "covar_cols":      covar_cols,
        "age_col":         age_col,
        "spline_df":       SPLINE_DF,
        "data_cols":       data_cols,
        "col_prefix":      col_prefix,
        "formula":         formula,
        "batch_col_names": list(batch_col_names),
        "variables":       model_vars,
    }
    joblib.dump(model_meta, out_model)

    print(f"Harmonized CSV : {out_csv}")
    print(f"Model          : {out_model}  ({len(data_cols)} variables)")


def main():
    parser = argparse.ArgumentParser(description="Univariate batch harmonization.")
    parser.add_argument("input_csv", help="path to input CSV")
    parser.add_argument("-o", "--output_dir", default=DEFAULTS["output_dir"],
                        help="output directory (default: ../../output/harmonization)")
    parser.add_argument("-p", "--out_prefix", default=DEFAULTS["out_prefix"],
                        help="output file prefix (default: input stem)")
    parser.add_argument("-c", "--col_prefix", default=DEFAULTS["col_prefix"],
                        help="prefix prepended to harmonized column names (default: none)")
    args = parser.parse_args()

    harm_univar(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        out_prefix=args.out_prefix or "",
        col_prefix=args.col_prefix,
    )


if __name__ == "__main__":
    main()
