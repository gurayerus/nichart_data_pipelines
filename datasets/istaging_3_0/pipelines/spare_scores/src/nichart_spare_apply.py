"""
NiChart_SPARE inference wrapper.

Builds and runs a NiChart_SPARE inference command from Python arguments.

Usage:
  python nichart_spare_apply.py input.csv \\
      --model   ./models/model_SPARE-AD.joblib \\
      --output  ./results/output_SPARE-AD.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "task":    "CL",
    "key_var": "MRID",
}


def call_apply(
    input_csv: Path,
    model_path: Path,
    output_csv: Path,
    task: str,
    key_var: str,
):
    cmd = [
        "NiChart_SPARE",
        "-a", "inference",
        "-t", task,
        "-i", str(input_csv),
        "-m", str(model_path),
        "-o", str(output_csv),
        "-kv", key_var,
    ]

    print("  CMD: " + " ".join(cmd))

    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        sys.exit("ERROR: NiChart_SPARE not found — is the package installed?")
    except Exception as e:
        sys.exit(f"ERROR running NiChart_SPARE inference: {e}")

    if r.returncode != 0:
        print("ERROR applying SPARE model:")
        print(r.stdout.decode("utf-8", errors="replace")[-2000:])
        print(r.stderr.decode("utf-8", errors="replace")[-2000:])
        sys.exit(1)
    print(f"  Output: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="NiChart_SPARE inference wrapper.")
    parser.add_argument("input_csv",
                        help="path to input CSV for inference")
    parser.add_argument("--model", required=True,
                        help="path to trained .joblib model")
    parser.add_argument("--output", required=True,
                        help="path for output CSV with SPARE scores")
    parser.add_argument("-t", "--task",    default=DEFAULTS["task"],
                        help="task type: CL or RG (default: CL)")
    parser.add_argument("--key_var",       default=DEFAULTS["key_var"],
                        help="key/ID column name (default: MRID)")
    args = parser.parse_args()

    input_csv  = Path(args.input_csv)
    model_path = Path(args.model)
    output_csv = Path(args.output)

    for p, label in [(input_csv, "input CSV"), (model_path, "model")]:
        if not p.exists():
            sys.exit(f"ERROR: {label} not found: {p}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    call_apply(
        input_csv=input_csv,
        model_path=model_path,
        output_csv=output_csv,
        task=args.task,
        key_var=args.key_var,
    )


if __name__ == "__main__":
    main()
