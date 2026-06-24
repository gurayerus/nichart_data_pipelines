"""
ComBat batch harmonization — apply trained model to new data.

Verifies that the test CSV contains all columns recorded during training
(batch, covariates, features) before building and running the R command.

Usage:
  python combat_apply.py input.csv --col_meta models/combat_train_cols.json
                                   --model    models/istag_h2dlmuse_train_model.rds
                                   -o         ./output
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


def load_col_metadata(meta_path: Path) -> dict:
    with meta_path.open(encoding="utf-8") as f:
        return json.load(f)


def read_header(input_csv: Path) -> list[str]:
    with input_csv.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def verify_columns(header: list[str], meta: dict) -> None:
    """Error-exit if any required column from training metadata is absent."""
    required = [meta["batch"]] + meta["covars"] + meta["features"]
    missing = [c for c in required if c not in header]
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} column(s) from training data not found in test CSV:\n"
            f"  {missing}"
        )


def detect_col_indices(header: list[str], meta: dict) -> dict:
    """Return 1-based column indices matching the training column layout."""
    idx = {c: header.index(c) + 1 for c in header}
    feat_ind = [idx[c] for c in meta["features"]]
    return {
        "batch_ind":  idx[meta["batch"]],
        "cov_ind":    [idx[c] for c in meta["covars"]],
        "age_ind":    idx[meta["covars"][0]],
        "feat_range": f"{min(feat_ind)}-{max(feat_ind)}",
    }


def call_apply(input_csv: Path, model_rds: Path, col_indices: dict, output_dir: Path):
    COMBAT_R = "/gpfs/fs001/cbica/home/harmang/micromamba/envs/combatfamqc/lib/R/library/ComBatFamQC/combatQC_CLI.R"
    MICRO = os.path.expanduser("~/bin/bin/micromamba")
    R_CMD = "{} run -n combatfamqc Rscript {}".format(MICRO, COMBAT_R)

    out_csv = os.path.join(output_dir, "results", "combat_harmonized.csv")

    cmd = (
        "{rscript} {input} "
        "--features {feat} "
        "-c {cov} "
        "-b {batch} "
        "-m gam "
        "-s {age} "
        "--predict TRUE "
        "--object {model} "
        "-d FALSE "
        "-v FALSE "
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
    print("  CMD: {}".format(cmd))

    try:
        r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        sys.exit(f"Error running ComBat R script: {e}")

    print("  Output: {}".format(out_csv))


def main():
    parser = argparse.ArgumentParser(description="ComBat batch harmonization — apply model.")
    parser.add_argument("input_csv", help="path to test CSV (col 1 must be MRID)")
    parser.add_argument("--col_meta", required=True, help="path to combat_train_cols.json written during training")
    parser.add_argument("--model", required=True, help="path to trained .rds model file")
    parser.add_argument("-o", "--output_dir", default=DEFAULTS["output_dir"], help="output directory (default: ../../output/harmonization)")
    args = parser.parse_args()

    input_csv  = Path(args.input_csv)
    model_rds  = Path(args.model)
    output_dir = Path(args.output_dir)
    meta_path  = Path(args.col_meta)

    # for p, label in [(input_csv, "input CSV"), (meta_path, "col metadata"), (model_rds, "model")]:
    #     if not p.exists():
    #         sys.exit(f"ERROR: {label} not found: {p}")

    meta   = load_col_metadata(meta_path)
    header = read_header(input_csv)

    verify_columns(header, meta)

    col_indices = detect_col_indices(header, meta)

    print(f"  batch   : {meta['batch']!r}  → col {col_indices['batch_ind']}")
    print(f"  covars  : {meta['covars']}  → cols {col_indices['cov_ind']}")
    print(f"  features: {len(meta['features'])} columns  → cols {col_indices['feat_range']}")

    call_apply(input_csv, model_rds, col_indices, output_dir)


if __name__ == "__main__":
    main()
