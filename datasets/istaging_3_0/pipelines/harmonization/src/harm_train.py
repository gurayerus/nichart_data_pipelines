"""
Univariate batch harmonization — training.

Fits OLS batch-harmonization models and removes batch effects from training data.

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
  <prefix>.joblib   — model: parameters + patsy DesignInfo for test-time use

Usage:
  python harm_train.py input.csv
  python harm_train.py input.csv -o ../output/harmonization -p myrun -c H_
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
    "model_dir":  None,
    "out_prefix": None,
    "col_prefix": "",
}


def harm_train(input_csv: Path, output_dir: Path, model_dir: Path | None = None,
               out_prefix: str = "", col_prefix: str = ""):
    model_dir = model_dir or output_dir
    prefix    = out_prefix or input_csv.stem
    out_csv   = output_dir / f"{prefix}.csv"
    out_model = model_dir  / f"{prefix}_model.joblib"

    if out_csv.exists() and out_model.exists():
        print(f"Skipping: output files already exist ({out_csv.name}, {out_model.name}).")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        sys.exit(f"Error reading input CSV '{input_csv}': {e}")

    all_cols = df.columns.tolist()

    if len(all_cols) < 6:
        sys.exit(f"Need at least 6 columns (MRID, batch, 3 covariates, 1+ data vars), got {len(all_cols)}.")

    mrid_col   = all_cols[0]
    batch_col  = all_cols[1]
    covar_cols = all_cols[2:5]
    data_cols  = all_cols[5:]
    age_col    = covar_cols[0]

    for c in covar_cols:
        df[c] = df[c].replace({"M": 1, "F": 0})

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

    # Compute spline knots from training age values so they can be saved and
    # reused at test time (avoids pickling patsy DesignInfo, which is unsupported).
    age_vals     = df_safe[s_age].values
    n_inner      = SPLINE_DF - 2
    inner_knots  = np.percentile(age_vals, np.linspace(0, 100, n_inner + 2)[1:-1]).tolist()
    spline_lower = float(age_vals.min())
    spline_upper = float(age_vals.max())

    knots_str     = repr([round(k, 8) for k in inner_knots])
    spline_term   = (f"cr({s_age}, knots={knots_str}, "
                     f"lower_bound={spline_lower}, upper_bound={spline_upper})")
    cov_terms     = [spline_term] + s_covars
    formula       = f"C({s_batch}) + " + " + ".join(cov_terms)
    covar_formula = " + ".join(cov_terms)   # no batch term; used for unknown-batch estimation

    # Build design matrices once (shared across all variables)
    X_dm       = dmatrix(formula, df_safe)
    X          = np.asarray(X_dm)
    X_covar_dm = dmatrix(covar_formula, df_safe)

    # Locate batch term slice in full design matrix
    batch_slice = None
    for term, slc in X_dm.design_info.term_slices.items():
        if s_batch in str(term):
            batch_slice = slc
            break

    if batch_slice is None:
        sys.exit(f"Could not locate batch term '{batch_col}' in design matrix.")

    X_batch         = X[:, batch_slice]
    batch_col_names = X_dm.design_info.column_names[batch_slice]

    # Indices of covariate columns in the full design matrix (everything except batch terms)
    batch_idxs = set(range(batch_slice.start, batch_slice.stop))
    covar_idxs = [i for i in range(X.shape[1]) if i not in batch_idxs]
    full_col_names = list(X_dm.design_info.column_names)

    # ── fit + harmonize each data variable ────────────────────────────────────
    harm_data  = {}
    model_vars = {}

    for var, s_var in zip(data_cols, [safe[c] for c in data_cols]):
        y   = df_safe[s_var].values.astype(float)
        fit = OLS(y, X).fit()

        beta_batch   = fit.params[batch_slice]
        batch_effect = X_batch @ beta_batch
        y_harm       = y - batch_effect

        out_col = f"{col_prefix}{var}" if col_prefix else var
        harm_data[out_col] = y_harm
        model_vars[var] = {
            "params":      dict(zip(full_col_names, fit.params)),
            "batch_terms": dict(zip(batch_col_names, beta_batch)),
            # positional covariate betas aligned to covar_formula column order
            "beta_covar":  fit.params[covar_idxs],
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
        "input_csv":        str(input_csv),
        "batch_col":        batch_col,
        "covar_cols":       covar_cols,
        "age_col":          age_col,
        "spline_df":        SPLINE_DF,
        "data_cols":        data_cols,
        "col_prefix":       col_prefix,
        "formula":          formula,
        # covar_formula has explicit knots — used by harm_test.py to rebuild the
        # covariate design matrix for unseen batches without needing DesignInfo
        "covar_formula":    covar_formula,
        "spline_knots":     inner_knots,
        "spline_lower":     spline_lower,
        "spline_upper":     spline_upper,
        "batch_col_names":  list(batch_col_names),
        "training_batches": sorted(df[batch_col].unique().tolist()),
        "variables":        model_vars,
    }
    joblib.dump(model_meta, out_model)

    print(f"Harmonized CSV : {out_csv}")
    print(f"Model          : {out_model}  ({len(data_cols)} variables)")


def main():
    parser = argparse.ArgumentParser(description="Univariate batch harmonization — training.")
    parser.add_argument("input_csv", help="path to input CSV")
    parser.add_argument("-o", "--output_dir", default=DEFAULTS["output_dir"],
                        help="output directory for harmonized CSV (default: ../../output/harmonization)")
    parser.add_argument("-m", "--model_dir",  default=DEFAULTS["model_dir"],
                        help="output directory for .joblib model (default: same as --output_dir)")
    parser.add_argument("-p", "--out_prefix", default=DEFAULTS["out_prefix"],
                        help="output file prefix (default: input stem)")
    parser.add_argument("-c", "--col_prefix", default=DEFAULTS["col_prefix"],
                        help="prefix prepended to harmonized column names (default: none)")
    args = parser.parse_args()

    harm_train(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        model_dir=Path(args.model_dir) if args.model_dir else None,
        out_prefix=args.out_prefix or "",
        col_prefix=args.col_prefix,
    )


if __name__ == "__main__":
    main()
