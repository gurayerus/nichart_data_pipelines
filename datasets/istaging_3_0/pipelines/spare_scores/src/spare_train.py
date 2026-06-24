"""
SPARE model training — linear SVM with n-fold cross-validation.

Input CSV column layout:
  col 1  : MRID (key)
  col 2  : target variable — categorical string/int → classification (CL)
                             continuous float           → regression   (RG)
  col 3+ : input features

Outputs:
  <prefix>_train_scores.csv   — MRID, SPARE  (cross-validated predictions)
  <prefix>_train_metrics.json — CV metrics
  <prefix>_model.joblib       — Pipeline(StandardScaler, SVM) trained on all samples

Task auto-detection: numeric target with > 10 unique values → RG, else → CL.
Override with --task {cl,rg}.

For CL, SPARE = decision-function distance from hyperplane (binary) or
predicted class label (multi-class).
For RG, SPARE = predicted value.

Usage:
  python spare_train.py input.csv
  python spare_train.py input.csv -o ../output -p myrun -n 5 -t cl
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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "output_dir":   THIS_DIR / "../../output/spare_scores",
    "model_dir":    None,
    "out_prefix":   None,
    "n_folds":      10,
    "task":         "auto",
    "class_weight": "balanced",
    "max_iter":     10_000,
}


def _detect_task(y: pd.Series) -> str:
    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.isna().any():
        return "cl"
    if numeric.nunique() <= 10:
        return "cl"
    return "rg"


def _cl_metrics(y_true, scores, labels) -> dict:
    is_binary = len(labels) == 2
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
        preds = scores  # already class labels for multi-class predict
        return {
            "accuracy":          float(accuracy_score(y_true, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        }


def _rg_metrics(y_true, preds) -> dict:
    corr, _ = pearsonr(y_true, preds)
    return {
        "rmse":      float(np.sqrt(mean_squared_error(y_true, preds))),
        "mae":       float(mean_absolute_error(y_true, preds)),
        "r2":        float(r2_score(y_true, preds)),
        "pearson_r": float(corr),
    }


def spare_train(
    input_csv: Path,
    output_dir: Path,
    model_dir: Path | None = None,
    out_prefix: str = "",
    n_folds: int = 10,
    task: str = "auto",
    class_weight: str = "balanced",
    max_iter: int = 10_000,
):
    model_dir   = model_dir or output_dir
    prefix      = out_prefix or input_csv.stem
    out_scores  = output_dir / f"{prefix}.csv"
    out_metrics = output_dir / f"{prefix}_metrics.json"
    out_model   = model_dir  / f"{prefix}_model.joblib"

    if out_scores.exists() and out_metrics.exists() and out_model.exists():
        print(f"Skipping: output files already exist for prefix '{prefix}'.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        sys.exit(f"Error reading '{input_csv}': {e}")

    cols       = df.columns.tolist()
    mrid_col   = cols[0]
    target_col = cols[1]
    feat_cols  = cols[2:]

    if not feat_cols:
        sys.exit("Need at least one feature column (col 3+).")

    for c in feat_cols:
        df[c] = df[c].replace({"M": 1, "F": 0})

    n_before = len(df)
    df = df.dropna(subset=[target_col] + feat_cols).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Warning: dropped {n_before - len(df)} rows with missing values.", file=sys.stderr)

    X     = df[feat_cols].values.astype(float)
    y_raw = df[target_col]

    # ── task detection ────────────────────────────────────────────────────────
    if task == "auto":
        task = _detect_task(y_raw)
        print(f"Auto-detected task: {task.upper()}")

    if task == "cl":
        if pd.to_numeric(y_raw, errors="coerce").isna().any():
            label_names = sorted(y_raw.unique().tolist())
            y = y_raw.map({v: i for i, v in enumerate(label_names)}).values.astype(int)
        else:
            y           = y_raw.values.astype(int)
            label_names = [int(v) for v in sorted(np.unique(y).tolist())]
        labels    = np.unique(y)
        is_binary = len(labels) == 2
    else:
        y           = y_raw.values.astype(float)
        label_names = None
        labels      = None
        is_binary   = None

    print(f"Input : {input_csv.name}  ({len(df)} samples, {len(feat_cols)} features)")
    if task == "cl":
        print(f"Task  : CL  |  target: '{target_col}'  |  classes: {label_names}")
    else:
        print(f"Task  : RG  |  target: '{target_col}'  |  range [{y.min():.3g}, {y.max():.3g}]")

    # ── cross-validation ──────────────────────────────────────────────────────
    if task == "cl":
        cv   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        svm  = LinearSVC(C=1.0, class_weight=class_weight, max_iter=max_iter, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("svm", svm)])
        cv_scores = cross_val_predict(
            pipe, X, y, cv=cv,
            method="decision_function" if is_binary else "predict",
        )
    else:
        cv   = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        svm  = LinearSVR(C=1.0, max_iter=max_iter)
        pipe = Pipeline([("scaler", StandardScaler()), ("svm", svm)])
        cv_scores = cross_val_predict(pipe, X, y, cv=cv)

    # ── CV metrics ────────────────────────────────────────────────────────────
    if task == "cl":
        metrics = _cl_metrics(y, cv_scores, labels)
    else:
        metrics = _rg_metrics(y, cv_scores)

    metrics.update({"n_samples": len(df), "n_features": len(feat_cols), "n_folds": n_folds, "task": task})
    print(f"CV metrics: { {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()} }")

    # ── fit final model on all training data ──────────────────────────────────
    pipe.fit(X, y)

    # ── write outputs ─────────────────────────────────────────────────────────
    pd.DataFrame({mrid_col: df[mrid_col], "SPARE": cv_scores}).to_csv(out_scores, index=False)

    with out_metrics.open("w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({
        "task":        task,
        "target_col":  target_col,
        "feat_cols":   feat_cols,
        "label_names": label_names,
        "labels":      labels.tolist() if labels is not None else None,
        "is_binary":   is_binary,
        "pipeline":    pipe,
    }, out_model)

    print(f"Scores  : {out_scores}")
    print(f"Metrics : {out_metrics}")
    print(f"Model   : {out_model}")


def main():
    parser = argparse.ArgumentParser(description="Train a SPARE model (linear SVM, n-fold CV).")
    parser.add_argument("input_csv",             help="path to training CSV")
    parser.add_argument("-o", "--output_dir",    default=DEFAULTS["output_dir"],
                        help="output directory for scores/metrics CSVs")
    parser.add_argument("-m", "--model_dir",     default=DEFAULTS["model_dir"],
                        help="output directory for .joblib model (default: same as --output_dir)")
    parser.add_argument("-p", "--out_prefix",    default=DEFAULTS["out_prefix"])
    parser.add_argument("-n", "--n_folds",       default=DEFAULTS["n_folds"], type=int)
    parser.add_argument("-t", "--task",          default=DEFAULTS["task"], choices=["auto", "cl", "rg"])
    parser.add_argument("-w", "--class_weight",  default=DEFAULTS["class_weight"],
                        choices=["balanced", "none"],
                        help="class weighting for CL (default: balanced)")
    parser.add_argument("--max_iter",            default=DEFAULTS["max_iter"], type=int)
    args = parser.parse_args()

    spare_train(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        model_dir=Path(args.model_dir) if args.model_dir else None,
        out_prefix=args.out_prefix or "",
        n_folds=args.n_folds,
        task=args.task,
        class_weight=args.class_weight if args.class_weight != "none" else None,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
