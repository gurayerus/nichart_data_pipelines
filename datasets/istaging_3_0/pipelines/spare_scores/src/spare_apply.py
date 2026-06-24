"""
SPARE model inference — apply a trained model to a new dataset.

Input CSV column layout (same as training):
  col 1  : MRID (key)
  col 2  : target variable (optional — if present and non-empty, metrics are computed)
  col 3+ : input features (must include all columns the model was trained on)

Outputs:
  <prefix>_test_scores.csv    — MRID, SPARE  (predicted scores)
  <prefix>_test_metrics.json  — evaluation metrics (only if ground-truth target available)

Usage:
  python spare_test.py input.csv model.joblib
  python spare_test.py input.csv model.joblib -o ../output -p mytest
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "output_dir": THIS_DIR / "../../output/spare_scores",
    "out_prefix": None,
}


def _cl_metrics(y_true, scores, labels, is_binary) -> dict:
    if is_binary:
        preds = (scores >= 0).astype(int)
        pos, neg = labels[1], labels[0]
        tp = int(((preds == 1) & (y_true == pos)).sum())
        tn = int(((preds == 0) & (y_true == neg)).sum())
        fp = int(((preds == 1) & (y_true == neg)).sum())
        fn = int(((preds == 0) & (y_true == pos)).sum())
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        try:
            auc = float(roc_auc_score(y_true == pos, scores))
        except Exception:
            auc = float("nan")
        return {
            "accuracy":          float(accuracy_score(y_true, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
            "auc":               auc,
            "sensitivity":       float(sens),
            "specificity":       float(spec),
        }
    else:
        return {
            "accuracy":          float(accuracy_score(y_true, scores)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, scores)),
        }


def _rg_metrics(y_true, preds) -> dict:
    corr, _ = pearsonr(y_true, preds)
    return {
        "rmse":      float(np.sqrt(mean_squared_error(y_true, preds))),
        "mae":       float(mean_absolute_error(y_true, preds)),
        "r2":        float(r2_score(y_true, preds)),
        "pearson_r": float(corr),
    }


def spare_test(
    input_csv: Path,
    model_path: Path,
    output_dir: Path,
    out_prefix: str = "",
):
    prefix      = out_prefix or input_csv.stem
    out_scores  = output_dir / f"{prefix}.csv"
    out_metrics = output_dir / f"{prefix}_metrics.json"

    if out_scores.exists():
        print(f"Skipping: {out_scores.name} already exists.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_meta = joblib.load(model_path)
    except Exception as e:
        sys.exit(f"Error loading model '{model_path}': {e}")

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        sys.exit(f"Error reading '{input_csv}': {e}")

    cols       = df.columns.tolist()
    mrid_col   = cols[0]
    target_col = cols[1] if len(cols) > 1 else None

    task      = model_meta["task"]
    feat_cols = model_meta["feat_cols"]
    labels    = model_meta["labels"]
    is_binary = model_meta["is_binary"]
    pipe      = model_meta["pipeline"]

    # Validate features
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        sys.exit(f"Test CSV is missing {len(missing)} feature column(s): {missing[:5]}"
                 + (" ..." if len(missing) > 5 else ""))

    for c in feat_cols:
        df[c] = df[c].replace({"M": 1, "F": 0})

    # Drop rows with missing feature values
    n_before = len(df)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Warning: dropped {n_before - len(df)} rows with missing feature values.", file=sys.stderr)

    X = df[feat_cols].values.astype(float)

    print(f"Input : {input_csv.name}  ({len(df)} samples, {len(feat_cols)} features)")
    print(f"Model : {model_path.name}  (task={task.upper()})")

    # ── inference ─────────────────────────────────────────────────────────────
    if task == "cl" and is_binary:
        scores = pipe.decision_function(X)
    elif task == "cl":
        scores = pipe.predict(X)
    else:
        scores = pipe.predict(X)

    pd.DataFrame({mrid_col: df[mrid_col], "SPARE": scores}).to_csv(out_scores, index=False)
    print(f"Scores : {out_scores}")

    # ── evaluation (only if ground-truth target is present and non-empty) ─────
    has_labels = (
        target_col is not None
        and target_col in df.columns
        and df[target_col].notna().any()
    )

    if has_labels:
        y_raw = df[target_col].dropna()

        # Restrict evaluation to labels seen during training.
        # Always use pd.to_numeric for matching: the column may be object dtype
        # when the CSV has mixed string/numeric values (e.g. "MCI" alongside "0","1"),
        # which would cause an int-keyed map lookup against string values to fail.
        if task == "cl" and labels is not None:
            train_label_set = set(labels)
            y_numeric  = pd.to_numeric(y_raw, errors="coerce")
            valid_mask = y_numeric.isin(train_label_set)
            n_excluded = int((~valid_mask).sum())
            if n_excluded:
                label_names = model_meta.get("label_names")
                print(f"  Excluding {n_excluded} rows with labels outside training set "
                      f"{label_names or sorted(train_label_set)}")
            if not valid_mask.any():
                sys.exit("No test samples with known training labels found.")
            y_raw  = y_raw[valid_mask]
            y_eval = y_numeric[valid_mask].values.astype(int)
        else:
            y_eval = y_raw.values.astype(float)

        df_eval = df.loc[y_raw.index]
        X_eval  = df_eval[feat_cols].values.astype(float)

        if task == "cl":
            eval_scores = (pipe.decision_function(X_eval) if is_binary
                           else pipe.predict(X_eval))
            metrics = _cl_metrics(y_eval, eval_scores, np.array(labels), is_binary)
        else:
            eval_preds = pipe.predict(X_eval)
            metrics = _rg_metrics(y_eval, eval_preds)

        metrics.update({"n_samples": len(df_eval), "task": task})
        print(f"Metrics: { {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()} }")

        with out_metrics.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics: {out_metrics}")


def main():
    parser = argparse.ArgumentParser(description="Apply a trained SPARE model to new data.")
    parser.add_argument("input_csv",          help="path to test CSV")
    parser.add_argument("model",              help="path to .joblib model from spare_train.py")
    parser.add_argument("-o", "--output_dir", default=DEFAULTS["output_dir"])
    parser.add_argument("-p", "--out_prefix", default=DEFAULTS["out_prefix"])
    args = parser.parse_args()

    spare_test(
        input_csv=Path(args.input_csv),
        model_path=Path(args.model),
        output_dir=Path(args.output_dir),
        out_prefix=args.out_prefix or "",
    )


if __name__ == "__main__":
    main()
