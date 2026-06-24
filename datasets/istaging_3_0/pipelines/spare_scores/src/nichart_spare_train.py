"""
NiChart_SPARE training wrapper.

Builds and runs a NiChart_SPARE trainer command from Python arguments.

Usage:
  python nichart_spare_train.py input.csv \\
      --model   ./models/model_SPARE-AD.joblib \\
      --target  DX_AD \\
      --ignore  Study
"""

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "output_dir":  THIS_DIR / "../../output/spare_scores",
    "model_dir":   None,
    "task":        "CL",
    "model_type":  "SVM",
    "kernel":      "linear_fast",
    "hypertuning": "True",
    "train_weights": "True",
    "n_folds":     5,
    "key_var":     "MRID",
    "ignore_col":  "Study",
    "class_balance": "True",
    "verbose":     1,
}


def call_train(
    input_csv: Path,
    model_path: Path,
    target_col: str,
    task: str,
    model_type: str,
    kernel: str,
    hypertuning: str,
    train_weights: str,
    n_folds: int,
    key_var: str,
    ignore_col: str,
    class_balance: str,
    verbose: int,
):
    cmd = [
        "NiChart_SPARE",
        "-a", "trainer",
        "-t", task,
        "-i", str(input_csv),
        "-mt", model_type,
        "-sk", kernel,
        "-ht", hypertuning,
        "-tw", train_weights,
        "-cf", str(n_folds),
        "-mo", str(model_path),
        "-kv", key_var,
        "-tc", target_col,
        "-ic", ignore_col,
        "-cb", class_balance,
        "-v",  str(verbose),
    ]

    print("  CMD: " + " ".join(cmd))

    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        sys.exit("ERROR: NiChart_SPARE not found — is the package installed?")
    except Exception as e:
        sys.exit(f"ERROR running NiChart_SPARE trainer: {e}")

    if r.returncode != 0:
        print("ERROR training SPARE model:")
        print(r.stdout.decode("utf-8", errors="replace")[-2000:])
        print(r.stderr.decode("utf-8", errors="replace")[-2000:])
        sys.exit(1)
    print(f"  Model saved: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="NiChart_SPARE training wrapper.")
    parser.add_argument("input_csv",
                        help="path to training CSV")
    parser.add_argument("--model", required=True,
                        help="output path for trained .joblib model")
    parser.add_argument("--target", required=True,
                        help="target column name (e.g. DX_AD)")
    parser.add_argument("-t",  "--task",          default=DEFAULTS["task"],
                        help="task type: CL (classification) or RG (regression) (default: CL)")
    parser.add_argument("--model_type",           default=DEFAULTS["model_type"],
                        help="model type passed to NiChart_SPARE (default: SVM)")
    parser.add_argument("--kernel",               default=DEFAULTS["kernel"],
                        help="SVM kernel (default: linear_fast)")
    parser.add_argument("--hypertuning",          default=DEFAULTS["hypertuning"],
                        help="enable hyperparameter tuning (default: True)")
    parser.add_argument("--train_weights",        default=DEFAULTS["train_weights"],
                        help="use training weights (default: True)")
    parser.add_argument("--n_folds",              default=DEFAULTS["n_folds"], type=int,
                        help="cross-validation folds (default: 5)")
    parser.add_argument("--key_var",              default=DEFAULTS["key_var"],
                        help="key/ID column name (default: MRID)")
    parser.add_argument("--ignore_col",           default=DEFAULTS["ignore_col"],
                        help="column to ignore during training (default: Study)")
    parser.add_argument("--class_balance",        default=DEFAULTS["class_balance"],
                        help="balance classes (default: True)")
    parser.add_argument("-v", "--verbose",        default=DEFAULTS["verbose"], type=int)
    args = parser.parse_args()

    input_csv  = Path(args.input_csv)
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        sys.exit(f"ERROR: input CSV not found: {input_csv}")

    call_train(
        input_csv=input_csv,
        model_path=model_path,
        target_col=args.target,
        task=args.task,
        model_type=args.model_type,
        kernel=args.kernel,
        hypertuning=args.hypertuning,
        train_weights=args.train_weights,
        n_folds=args.n_folds,
        key_var=args.key_var,
        ignore_col=args.ignore_col,
        class_balance=args.class_balance,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
