"""
Univariate batch harmonization — testing.

Applies a trained harmonization model to a new dataset.

  Known batches (seen during training):
    Subtract the saved batch-effect beta directly.

  Unseen batches:
    Estimate the batch effect as the mean covariate residual per new batch,
    then subtract.  Residual = y - X_covar @ beta_covar, where X_covar is
    built with the training spline knots (stored in the saved DesignInfo).

Input CSV must follow the same column layout as the training CSV:
  col 1  : MRID (key)
  col 2  : batch variable
  col 3-5: covariates (same order as training)
  col 6+ : data variables (same set as training)

Outputs:
  <prefix>.csv  — MRID + harmonized columns

Usage:
  python harm_test.py input.csv model.joblib
  python harm_test.py input.csv model.joblib -o ../output/harmonization -p mytest -c H_
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from patsy import dmatrix

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "output_dir": THIS_DIR / "../../output/harmonization",
    "out_prefix": None,
    "col_prefix": None,   # None → use model's stored col_prefix
}


def _batch_betas_from_terms(batch_terms: dict) -> dict:
    """Return {str(batch_value): beta} parsed from patsy-encoded batch_terms keys.

    Patsy encodes non-reference levels as 'C(_v1)[T.value]'; the reference
    level is absent (its beta is 0 by definition).
    """
    result = {}
    for key, val in batch_terms.items():
        if "[T." in key:
            result[key.split("[T.")[-1].rstrip("]")] = float(val)
    return result


def harm_test(
    input_csv: Path,
    model_path: Path,
    output_dir: Path,
    out_prefix: str = "",
    col_prefix: str | None = None,
):
    prefix  = out_prefix or input_csv.stem
    out_csv = output_dir / f"{prefix}.csv"

    if out_csv.exists():
        print(f"Skipping: output file already exists ({out_csv.name}).")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        sys.exit(f"Error loading model '{model_path}': {e}")

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        sys.exit(f"Error reading input CSV '{input_csv}': {e}")

    all_cols   = df.columns.tolist()
    mrid_col   = all_cols[0]
    batch_col  = all_cols[1]
    covar_cols = all_cols[2:5]
    data_cols  = all_cols[5:]

    train_data_cols = model["data_cols"]
    missing = [c for c in train_data_cols if c not in data_cols]
    if missing:
        sys.exit(
            f"Test CSV is missing {len(missing)} data column(s) from training: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    for c in covar_cols:
        df[c] = df[c].replace({"M": 1, "F": 0})
    for c in covar_cols + data_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=covar_cols + train_data_cols).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Warning: dropped {n_before - len(df)} rows with missing values.", file=sys.stderr)

    # Same positional safe-name scheme as training (col position → _v{i})
    safe    = {c: f"_v{i}" for i, c in enumerate(all_cols)}
    df_safe = df.rename(columns=safe)

    training_batches = set(model["training_batches"])
    batch_values     = df[batch_col].values

    known_mask   = np.array([b in training_batches for b in batch_values])
    unknown_mask = ~known_mask

    known_batches   = sorted({b for b in batch_values if     known_mask[batch_values == b].any()})
    unknown_batches = sorted({b for b in batch_values if not known_mask[batch_values == b].any()})

    print(f"Input : {input_csv.name}  ({len(df)} rows, {len(train_data_cols)} data vars)")
    print(f"  known batches   ({len(known_batches)}): {known_batches}")
    print(f"  unknown batches ({len(unknown_batches)}): {unknown_batches}")

    # Build covariate design matrix for unknown-batch rows once.
    # covar_formula has explicit spline knots from training, so dmatrix()
    # reproduces the same basis without needing a saved DesignInfo.
    X_covar_unknown = None
    if unknown_mask.any():
        df_safe_unknown = df_safe[unknown_mask].reset_index(drop=True)
        try:
            X_covar_unknown = np.asarray(dmatrix(model["covar_formula"], df_safe_unknown))
        except Exception as e:
            sys.exit(f"Error building covariate design matrix for unknown batches: {e}")

    _col_prefix = col_prefix if col_prefix is not None else model.get("col_prefix", "")

    harm_data = {}

    for var in train_data_cols:
        var_model  = model["variables"][var]
        y          = df[var].values.astype(float)
        y_harm     = y.copy()

        # ── known batches: look up saved batch effect ──────────────────────────
        batch_betas = _batch_betas_from_terms(var_model["batch_terms"])
        for b_str, beta in batch_betas.items():
            mask = np.array([str(bv) == b_str for bv in batch_values]) & known_mask
            y_harm[mask] -= beta
        # reference batch: no correction (beta = 0 by construction)

        # ── unknown batches: estimate batch effect from covariate residuals ────
        if unknown_mask.any():
            beta_covar_vec = np.asarray(var_model["beta_covar"])
            fitted_covar      = X_covar_unknown @ beta_covar_vec
            y_unknown_raw     = y[unknown_mask]
            residuals         = y_unknown_raw - fitted_covar
            unknown_batch_arr = batch_values[unknown_mask]

            for b in unknown_batches:
                mask_local = unknown_batch_arr == b
                beta_est   = residuals[mask_local].mean()
                y_harm[batch_values == b] -= beta_est

        out_col = f"{_col_prefix}{var}" if _col_prefix else var
        harm_data[out_col] = y_harm

    if unknown_batches:
        print(f"  Unknown-batch effects estimated per variable and subtracted.")

    # ── write output CSV (MRID + harmonized columns only) ─────────────────────
    out_df = pd.concat(
        [df[[mrid_col]], pd.DataFrame(harm_data, index=df.index)],
        axis=1,
    )
    out_df.to_csv(out_csv, index=False)
    print(f"Harmonized CSV : {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Univariate batch harmonization — testing.")
    parser.add_argument("input_csv",   help="path to input CSV")
    parser.add_argument("model",       help="path to .joblib model from harm_train.py")
    parser.add_argument("-o", "--output_dir", default=DEFAULTS["output_dir"],
                        help="output directory (default: ../../output/harmonization)")
    parser.add_argument("-p", "--out_prefix", default=DEFAULTS["out_prefix"],
                        help="output file prefix (default: input stem)")
    parser.add_argument("-c", "--col_prefix", default=DEFAULTS["col_prefix"],
                        help="prefix prepended to harmonized column names (default: from model)")
    args = parser.parse_args()

    harm_test(
        input_csv=Path(args.input_csv),
        model_path=Path(args.model),
        output_dir=Path(args.output_dir),
        out_prefix=args.out_prefix or "",
        col_prefix=args.col_prefix,
    )


if __name__ == "__main__":
    main()
