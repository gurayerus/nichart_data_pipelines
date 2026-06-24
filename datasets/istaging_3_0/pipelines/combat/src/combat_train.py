"""
ComBat batch harmonization — training.

Detects column indices from --covars and --batch names against the CSV header.
All columns other than MRID (col 1), covariates, and batch are treated as features.
The first covariate is assumed to be Age (gets a spline in the GAM model).

Outputs:
  <outdir>/models/istag_h2dlmuse_train_model.rds  — fitted ComBat model
  <outdir>/reference_harmonized_internal.csv       — harmonized training data

Usage:
  python combat_train.py input.csv -o ./output --covars Age,Sex,DLICV --batch Batch
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "output_dir": THIS_DIR / "../../output/harmonization",
}


def detect_col_indices(input_csv: Path, covar_names: list[str], batch_name: str) -> dict:
    """Read the CSV header and return 1-based column indices for batch, covars, and features."""
    with input_csv.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    # validate
    missing = [c for c in covar_names + [batch_name] if c not in header]
    if missing:
        sys.exit(f"ERROR: columns not found in {input_csv.name}: {missing}\n"
                 f"  Available: {header}")

    skip = {"MRID"} | set(covar_names) | {batch_name}
    feat_cols  = [c for c in header if c not in skip]
    if not feat_cols:
        sys.exit("ERROR: no feature columns remain after excluding MRID, covariates, and batch.")

    # 1-based indices (R convention)
    idx = {c: header.index(c) + 1 for c in header}
    feat_ind = [idx[c] for c in feat_cols]
    return {
        "batch_ind":  idx[batch_name],
        "cov_ind":    [idx[c] for c in covar_names],
        "age_ind":    idx[covar_names[0]],
        "feat_ind":   feat_ind,
        "feat_range": f"{min(feat_ind)}-{max(feat_ind)}",
        "feat_cols":  feat_cols,
    }


def write_col_metadata(col_indices: dict, covar_names: list[str], batch_name: str, output_dir: Path):
    """Write column metadata to <output_dir>/models/combat_train_cols.json for use at apply time."""
    meta = {
        "batch":    batch_name,
        "covars":   covar_names,
        "features": col_indices["feat_cols"],
    }
    out_path = output_dir / "models" / "combat_train_cols.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Col metadata: {out_path}")


def call_train(input_csv: Path, col_indices: dict, output_dir: Path):
    COMBAT_R = "/gpfs/fs001/cbica/home/harmang/micromamba/envs/combatfamqc/lib/R/library/ComBatFamQC/combatQC_CLI.R"
    MICRO = os.path.expanduser("~/bin/bin/micromamba")
    R_CMD = "{} run -n combatfamqc Rscript {}".format(MICRO, COMBAT_R)

    model_rds = os.path.join(output_dir, "models", "istag_h2dlmuse_train_model.rds")
    out_csv   = os.path.join(output_dir, "reference_harmonized_internal.csv")

    cmd_train = (
        "{rscript} {input} "
        "--features {feat} "
        "-c {cov} "
        "-b {batch} "
        "-m gam "
        "-s {age} "
        "-d FALSE "
        "-v FALSE "
        "--mout {model} "
        "--outdir {outdir}"
    ).format(
        rscript=R_CMD,
        input=input_csv,
        feat=col_indices["feat_range"],
        cov=",".join(str(i) for i in col_indices["cov_ind"]),
        batch=col_indices["batch_ind"],
        age=col_indices["age_ind"],
        model=model_rds,
        outdir=out_csv,
    )
    print("  CMD: {}".format(cmd_train))

    try:
        r = subprocess.run(cmd_train, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("  Model saved: {}".format(model_rds))
    except Exception as e:
        sys.exit(f"Error running ComBat R script: {e}")



def main():
    parser = argparse.ArgumentParser(description="ComBat batch harmonization — training.")
    parser.add_argument("input_csv",
                        help="path to input CSV (col 1 must be MRID)")
    parser.add_argument("-o", "--output_dir", default=DEFAULTS["output_dir"],
                        help="output directory (default: ../../output/harmonization)")
    parser.add_argument("--covars", required=True,
                        help="comma-separated covariate column names; first is treated as Age")
    parser.add_argument("--batch", required=True,
                        help="batch column name")
    args = parser.parse_args()

    input_csv  = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    covar_names = [c.strip() for c in args.covars.split(",")]

    col_indices = detect_col_indices(input_csv, covar_names, args.batch)

    print(f"  batch   : {args.batch!r}  → col {col_indices['batch_ind']}")
    print(f"  covars  : {covar_names}  → cols {col_indices['cov_ind']}")
    print(f"  features: {len(col_indices['feat_cols'])} columns  → cols {col_indices['feat_range']}")

    write_col_metadata(col_indices, covar_names, args.batch, output_dir)
    call_train(input_csv, col_indices, output_dir)


if __name__ == "__main__":
    main()
