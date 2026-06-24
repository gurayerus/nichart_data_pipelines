"""
Run all iSTAGING 3.0 data processing steps end-to-end.

Intermediate outputs are written under output/intermediate/<pipeline>/{data,results,models}/.

Steps:
  data_prep  / split_input         — split anonymized CSV into themed sub-tables
  data_prep  / calc_derived_rois   — compute derived H_DLMUSE ROIs
  harmonization                    — skipped unless data_anon/tasks/harmonization/ exists
     .A  data_selection / merge_files  — build harmonization train + test CSVs
     .B  harmonization / harm_train        — fit batch-effect models on training set
     .C  harmonization / harm_apply         — apply models to harmonization test CSV
  combat                           — skipped unless data_anon/tasks/combat/ exists
     .A  data_selection / merge_files  — build combat train + test CSVs
     .B  harmonization / harm_train        — fit batch-effect models on training set
     .C  harmonization / harm_apply         — apply models to combat test CSV
  spare_scores (sklearn)            — always runs
     .A  data_selection / merge_files  — build ML train + test CSVs
     .B  spare_scores / spare_train        — train SPARE model (linear SVM, 10-fold CV)
     .C  spare_scores / spare_apply        — apply SPARE model to test set
  spare_scores (NiChart_SPARE)      — always runs
     .A  data_selection / merge_files  — build ML train + test CSVs
     .B  nichart_spare_train               — train via NiChart_SPARE trainer
     .C  nichart_spare_apply               — inference via NiChart_SPARE

Usage:
  python run_pipelines.py
  python run_pipelines.py --input_csv <path>
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

_UTILS = Path(__file__).resolve().parent / "utils"
sys.path.insert(0, str(_UTILS))
from logger import setup_logger  # type: ignore[import]

THIS_DIR   = Path(__file__).parent
INPUT_DIR  = None  # set by --input_dir
OUTPUT_DIR = None  # set by --output_dir

PYTHON = sys.executable

TOTAL_STEPS = 6

log = logging.getLogger("run_pipelines")


class _StepCounter:
    """Auto-incrementing step counter with optional lettered sub-steps."""
    def __init__(self):
        self.n = 0
        self._sub = 0

    def next(self) -> int:
        self.n += 1
        self._sub = 0
        return self.n

    def sub(self) -> str:
        self._sub += 1
        return f"{self.n}.{chr(64 + self._sub)}"


STEP = _StepCounter()



def step_header(n: int | str, label: str) -> None:
    log.info(f"\n[STEP {n}/{TOTAL_STEPS}]  {label}")
    log.debug("=" * 72)


def symlink(src: Path, dst: Path) -> None:
    """Create a symlink at dst -> src, falling back to copy on Windows without privilege."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        log.debug(f"  Skipping link: {dst.name} already exists.")
        return
    rel = Path(os.path.relpath(src, dst.parent))
    try:
        dst.symlink_to(rel)
        log.debug(f"  Linked  {dst.relative_to(OUTPUT_DIR)}  ->  {rel}")
    except OSError:
        shutil.copy2(src, dst)
        log.debug(f"  Copied  {dst.relative_to(OUTPUT_DIR)}  <-  {src.name}")


def _read_target(task_dir: Path, desc_file: str = "training_data_desc.json") -> str:
    with (task_dir / desc_file).open(encoding="utf-8") as f:
        return json.load(f)["target"]


def run(label: str, cmd: list) -> None:
    log.info(f"  » {label}")
    log.debug("    CMD: " + " ".join(str(c) for c in cmd))

    result = subprocess.run(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in (result.stdout or "").splitlines():
        log.debug("    " + line)

    if result.returncode != 0:
        # Show last 20 lines on console so the user can see what went wrong
        tail = (result.stdout or "").strip().splitlines()[-20:]
        for line in tail:
            log.error("    " + line)
        log.error(f"  FAILED: {label}  (exit {result.returncode})")
        sys.exit(1)

    for line in (result.stdout or "").splitlines():
        if "already exists" in line:
            log.info("    " + line)

    log.info(f"\n  SUCCESS: {label}")


def main():
    parser = argparse.ArgumentParser(description="Run full iSTAGING 3.0 processing pipeline.")
    parser.add_argument("--input_dir",  required=True,
                        help="anonymized input directory")
    parser.add_argument("--output_dir", required=True,
                        help="output directory root")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show debug output (commands, subprocess output) on the console")
    args = parser.parse_args()

    global INPUT_DIR, OUTPUT_DIR
    INPUT_DIR  = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)

    csv_candidates = list((INPUT_DIR / "data").glob("*.csv"))
    if len(csv_candidates) == 0:
        sys.exit(f"Error: no CSV file found in {INPUT_DIR / 'data'}")
    if len(csv_candidates) > 1:
        sys.exit(f"Error: multiple CSV files found in {INPUT_DIR / 'data'} — "
                 f"{[f.name for f in csv_candidates]}")
    input_csv = csv_candidates[0]
    log.info(f"  Detected input CSV: {input_csv}")

    # ── task lists — set any to [] to skip that step ──────────────────────────
    SPLIT_TASKS    = ["split"]

    CALC_TASKS     = ["calc"]
    HARM_TASKS     = ["harmonization"]
    COMBAT_TASKS   = ["combat"]
    SPARE_TASKS    = ["spare-ad-raw", "spare-ad-h", "spare-ad-h2"]
    NICHART_TASKS  = ["nichart-sparead-raw"]
    NICHART_KERNEL = "linear_fast"

    # CALC_TASKS     = []
    # HARM_TASKS     = []
    # COMBAT_TASKS   = []
    # SPARE_TASKS    = []
    # NICHART_TASKS  = []

    # SPARE_TASKS    = ["spare-ad-raw"]

    # ── logging ───────────────────────────────────────────────────────────────
    setup_logger("run_pipelines", verbose=args.verbose, log_dir=OUTPUT_DIR / "logs")
    log.debug("Command: " + " ".join(sys.argv))

    # ── output subdirectories ─────────────────────────────────────────────────
    for subdir in ("final/data", "final/models", "intermediate"):
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # ── split input CSV into sub-tables ───────────────────────────────────────
    STEP.next()
    if SPLIT_TASKS:
        step_header(STEP.n, "data_prep / split_input into sub-tables")
        run("data_prep / split_input", [
            PYTHON, THIS_DIR / "data_utils/data_prep/src/split_input.py",
            "-i", input_csv,
            "-d", INPUT_DIR / "dictionaries/dict_data_files.json",
            "-v", INPUT_DIR / "dictionaries/dict_var_groups.json",
            "-o", OUTPUT_DIR / "final" / "data",
        ])
    else:
        log.info(f"  Skipping step {STEP.n} (split_input)")

    # ── calculate derived H_DLMUSE ROIs ───────────────────────────────────────
    STEP.next()
    if CALC_TASKS:
        step_header(STEP.n, "data_prep / calc_derived_rois — compute derived H_DLMUSE ROIs")
        run("data_prep / calc_derived_rois", [
            PYTHON, THIS_DIR / "data_utils/data_prep/src/calc_derived_rois.py",
            "-i", OUTPUT_DIR / "final" / "data" / "istag_hdlmuse_missingvars.csv",
            "-m", INPUT_DIR / "dictionaries/muse_mapping_derived.csv",
            "-o", OUTPUT_DIR / "final" / "data" / "istag_hdlmuse.csv",
        ])
    else:
        log.info(f"  Skipping step {STEP.n} (calc_derived_rois)")

    # ── harmonization ─────────────────────────────────────────────────────────
    STEP.next()
    if HARM_TASKS:
        HARM_TASK_DIR = INPUT_DIR / "tasks" / HARM_TASKS[0]
        HARM_DIR      = OUTPUT_DIR / "intermediate" / HARM_TASKS[0]
        if not HARM_TASK_DIR.exists():
            log.warning(f"  Skipping step {STEP.n} ({HARM_TASKS[0]}) — folder not found: {HARM_TASK_DIR}")
        else:
            step_header(STEP.sub(), "harmonization / merge_files — build train + test CSVs")
            for desc_file in ("training_data_desc.json", "testing_data_desc.json"):
                run(f"merge_files ({desc_file})", [
                    PYTHON, THIS_DIR / "data_utils/data_selection/src/merge_files.py",
                    "--task_dir",   HARM_TASK_DIR,
                    "--desc_file",  desc_file,
                    "--var_groups", INPUT_DIR / "dictionaries/dict_var_groups.json",
                    "--data_files", INPUT_DIR / "dictionaries/dict_data_files.json",
                    "--root",        INPUT_DIR.parent,
                    "--output_root", OUTPUT_DIR,
                ])

            step_header(STEP.sub(), "harmonization / harm_train — fit batch-effect models")
            run("harm_train", [
                PYTHON, THIS_DIR / "harmonization/src/harm_train.py",
                HARM_DIR / "training_data.csv",
                "--output_dir", HARM_DIR / "results",
                "--model_dir",  HARM_DIR / "models",
                "--out_prefix", "istag_h2dlmuse_train",
                "--col_prefix", "H2_",
            ])
            symlink(HARM_DIR / "models" / "istag_h2dlmuse_train_model.joblib",
                    OUTPUT_DIR / "final" / "models" / "istag_h2dlmuse_train_model.joblib")

            step_header(STEP.sub(), "harmonization / harm_apply — apply models to test CSV")
            HARM_MODEL = HARM_DIR / "models" / "istag_h2dlmuse_train_model.joblib"
            run("harm_apply", [
                PYTHON, THIS_DIR / "harmonization/src/harm_apply.py",
                HARM_DIR / "testing_data.csv",
                HARM_MODEL,
                "--output_dir", HARM_DIR / "results",
                "--out_prefix", "istag_h2dlmuse",
            ])
            symlink(HARM_DIR / "results" / "istag_h2dlmuse.csv",
                    OUTPUT_DIR / "final" / "data" / "istag_h2dlmuse.csv")
    else:
        log.info(f"  Skipping step {STEP.n} (harmonization)")

    # ── combat ────────────────────────────────────────────────────────────────
    STEP.next()
    if COMBAT_TASKS:
        COMBAT_TASK_DIR = INPUT_DIR / "tasks" / COMBAT_TASKS[0]
        COMBAT_DIR      = OUTPUT_DIR / "intermediate" / COMBAT_TASKS[0]
        if not COMBAT_TASK_DIR.exists():
            log.warning(f"  Skipping step {STEP.n} ({COMBAT_TASKS[0]}) — folder not found: {COMBAT_TASK_DIR}")
        else:
            step_header(STEP.sub(), "combat / merge_files — build train + test CSVs")
            for desc_file in ("training_data_desc.json", "testing_data_desc.json"):
                run(f"merge_files ({desc_file})", [
                    PYTHON, THIS_DIR / "data_utils/data_selection/src/merge_files.py",
                    "--task_dir",   COMBAT_TASK_DIR,
                    "--desc_file",  desc_file,
                    "--var_groups", INPUT_DIR / "dictionaries/dict_var_groups.json",
                    "--data_files", INPUT_DIR / "dictionaries/dict_data_files.json",
                    "--root",        INPUT_DIR.parent,
                    "--output_root", OUTPUT_DIR,
                ])

            step_header(STEP.sub(), "combat / combat_train — fit batch-effect models")
            run("combat_train", [
                PYTHON, THIS_DIR / "combat/src/combat_train.py",
                COMBAT_DIR / "training_data.csv",
                "-o",       COMBAT_DIR,
                "--covars", "Age,Sex,DLMUSE_702",
                "--batch",  "batch",
            ])
            # symlink(COMBAT_DIR / "models" / "istag_h2dlmuse_train_model.rds",
            #         OUTPUT_DIR / "final" / "models" / "istag_h2dlmuse_train_model.rds")

            step_header(STEP.sub(), "combat / combat_apply — apply models to test CSV")
            run("combat_apply", [
                PYTHON, THIS_DIR / "combat/src/combat_apply.py",
                COMBAT_DIR / "testing_data.csv",
                "--col_meta", COMBAT_DIR / "models" / "combat_train_cols.json",
                "--model",    COMBAT_DIR / "models" / "istag_h2dlmuse_train_model.rds",
                "-o",         COMBAT_DIR,
            ])
            # symlink(COMBAT_DIR / "results" / "combat_harmonized.csv",
            #         OUTPUT_DIR / "final" / "data" / "istag_h2dlmuse.csv")
    else:
        log.info(f"  Skipping step {STEP.n} (combat)")

    # ── spare scores (sklearn) ────────────────────────────────────────────────
    SPARE_OUT_DIR = OUTPUT_DIR / "intermediate" / "spare_scores"
    STEP.next()
    if SPARE_TASKS:
        step_header(STEP.sub(), f"spare_scores / merge_files — build train + test CSVs  {SPARE_TASKS}")
        for task in SPARE_TASKS:
            for desc_file in ("training_data_desc.json", "testing_data_desc.json"):
                run(f"merge_files ({task} / {desc_file})", [
                    PYTHON, THIS_DIR / "data_utils/data_selection/src/merge_files.py",
                    "--task_dir",   INPUT_DIR / "tasks" / task,
                    "--desc_file",  desc_file,
                    "--var_groups", INPUT_DIR / "dictionaries/dict_var_groups.json",
                    "--data_files", INPUT_DIR / "dictionaries/dict_data_files.json",
                    "--root",        INPUT_DIR.parent,
                    "--output_root", OUTPUT_DIR,
                ])

        step_header(STEP.sub(), f"spare_scores / spare_train — train SPARE models  {SPARE_TASKS}")
        for task in SPARE_TASKS:
            spare_dir = SPARE_OUT_DIR / task
            run(f"spare_train ({task})", [
                PYTHON, THIS_DIR / "spare_scores/src/spare_train.py",
                spare_dir / "training_data.csv",
                "--output_dir", spare_dir / "results",
                "--model_dir",  spare_dir / "models",
                "--out_prefix", f"istag_{task}_train",
            ])
        for task in SPARE_TASKS:
            symlink(SPARE_OUT_DIR / task / "models" / f"istag_{task}_train_model.joblib",
                    OUTPUT_DIR / "final" / "models" / f"istag_{task}_train_model.joblib")

        step_header(STEP.sub(), f"spare_scores / spare_apply — apply SPARE models  {SPARE_TASKS}")
        for task in SPARE_TASKS:
            spare_dir   = SPARE_OUT_DIR / task
            spare_model = spare_dir / "models" / f"istag_{task}_train_model.joblib"
            run(f"spare_apply ({task})", [
                PYTHON, THIS_DIR / "spare_scores/src/spare_apply.py",
                spare_dir / "testing_data.csv",
                spare_model,
                "--output_dir", spare_dir / "results",
                "--out_prefix", f"istag_{task}",
            ])
        for task in SPARE_TASKS:
            symlink(SPARE_OUT_DIR / task / "results" / f"istag_{task}.csv",
                    OUTPUT_DIR / "final" / "data" / f"istag_{task}.csv")
    else:
        log.info(f"  Skipping step {STEP.n} (spare_scores sklearn)")

    # ── nichart spare scores ──────────────────────────────────────────────────
    NICHART_OUT_DIR = OUTPUT_DIR / "intermediate" / "spare_scores"
    STEP.next()
    if NICHART_TASKS:
        step_header(STEP.sub(), f"nichart_spare / merge_files — build train + test CSVs  {NICHART_TASKS}")
        for task in NICHART_TASKS:
            for desc_file in ("training_data_desc.json", "testing_data_desc.json"):
                run(f"merge_files ({task} / {desc_file})", [
                    PYTHON, THIS_DIR / "data_utils/data_selection/src/merge_files.py",
                    "--task_dir",   INPUT_DIR / "tasks" / task,
                    "--desc_file",  desc_file,
                    "--var_groups", INPUT_DIR / "dictionaries/dict_var_groups.json",
                    "--data_files", INPUT_DIR / "dictionaries/dict_data_files.json",
                    "--root",        INPUT_DIR.parent,
                    "--output_root", OUTPUT_DIR,
                ])

        step_header(STEP.sub(), f"nichart_spare / nichart_spare_train — train SPARE models  {NICHART_TASKS}")
        for task in NICHART_TASKS:
            task_dir   = INPUT_DIR / "tasks" / task
            spare_dir  = NICHART_OUT_DIR / task
            model_path = spare_dir / "models" / f"model_{task}_{NICHART_KERNEL}.joblib"
            run(f"nichart_spare_train ({task})", [
                PYTHON, THIS_DIR / "spare_scores/src/nichart_spare_train.py",
                spare_dir / "training_data.csv",
                "--model",  model_path,
                "--target", _read_target(task_dir),
                "--kernel", NICHART_KERNEL,
            ])
        for task in NICHART_TASKS:
            model_path = NICHART_OUT_DIR / task / "models" / f"model_{task}_{NICHART_KERNEL}.joblib"
            symlink(model_path, OUTPUT_DIR / "final" / "models" / model_path.name)

        step_header(STEP.sub(), f"nichart_spare / nichart_spare_apply — apply SPARE models  {NICHART_TASKS}")
        for task in NICHART_TASKS:
            spare_dir  = NICHART_OUT_DIR / task
            model_path = spare_dir / "models" / f"model_{task}_{NICHART_KERNEL}.joblib"
            output_csv = spare_dir / "results" / f"output_{task}_{NICHART_KERNEL}.csv"
            run(f"nichart_spare_apply ({task})", [
                PYTHON, THIS_DIR / "spare_scores/src/nichart_spare_apply.py",
                spare_dir / "testing_data.csv",
                "--model",  model_path,
                "--output", output_csv,
            ])
        for task in NICHART_TASKS:
            output_csv = NICHART_OUT_DIR / task / "results" / f"output_{task}_{NICHART_KERNEL}.csv"
            symlink(output_csv, OUTPUT_DIR / "final" / "data" / output_csv.name)
    else:
        log.info(f"  Skipping step {STEP.n} (nichart_spare)")

    log.info("\nAll steps completed.")


if __name__ == "__main__":
    main()
