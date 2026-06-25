# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains Python data processing pipelines for neuroimaging datasets. Currently one dataset is implemented: `datasets/istaging_3_0/`, which processes the iSTAGING 3.0 multi-site brain MRI study data.

All pipeline scripts live under `datasets/istaging_3_0/pipelines/` and are run from that directory.

## Running the Pipeline

The end-to-end workflow is four sequential steps, each a standalone script:

```bash
cd datasets/istaging_3_0/pipelines

# Step 1: Anonymize raw private data
python s1_anonymize.py --in_dir <private_data_dir> --in_csv <main.csv> --out_dir <anon_dir>

# Step 2: (Optional) Subset to a sample
python s2_select_sample.py --in_dir <anon_dir> --in_sample_csv <sample/list.csv> --out_dir <sel_dir>

# Step 3: Run all processing pipelines (harmonization + SPARE scores)
python s3_run_pipelines.py --input_dir <sel_dir> --output_dir <out_dir> [-v]

# Step 4: Run QC plots for a specific task
python s4_run_qc.py --in_dir <sel_dir> --out_dir <out_dir> --task <task_name> [-v]
```

All four scripts accept `-v` / `--verbose` to show debug output and subprocess commands on the console. Logs are always written to `<out_dir>/logs/`.

The `example_run.sh` at `datasets/istaging_3_0/pipelines/example_run.sh` shows a concrete invocation sequence.

## Architecture

### Input layout (anonymized)
```
<input_dir>/
  data/               — anonymized CSV files (split by theme after step 1)
  samples/            — per-sample MRID list CSVs
  dictionaries/
    dict_data_files.json   — maps file_key → file_prefix (used by merge_files)
    dict_var_groups.json   — named column groups; supports concat and prefix derivation
    muse_mapping_derived.csv
  tasks/<task_name>/  — per-task JSON descriptors consumed by merge_files + s4_run_qc
```

### Output layout
```
<output_dir>/
  final/
    data/    — symlinks to key CSVs (harmonized data, SPARE scores)
    models/  — symlinks to trained .joblib model files
  intermediate/
    harmonization/    — harm_train / harm_apply working files
    combat/           — combat working files
    spare_scores/<task>/  — per-SPARE-task working files
  logs/      — timestamped log files
  qc/        — distribution plots and merged verification CSVs
```

### Sub-modules called by `s3_run_pipelines.py`

All sub-scripts are invoked as subprocesses; `s3_run_pipelines.py` orchestrates them in order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `data_utils/data_prep/src/split_input.py` | Split main CSV into themed sub-tables using `dict_data_files.json` |
| 2 | `data_utils/data_prep/src/calc_derived_rois.py` | Compute derived H_DLMUSE ROI columns from a MUSE mapping CSV |
| 3 | `data_utils/data_selection/src/merge_files.py` | Build train/test CSVs for harmonization (reads `training_data_desc.json` / `testing_data_desc.json`) |
| 4 | `harmonization/src/harm_train.py` | OLS batch-effect model fitting; saves `.joblib` model |
| 5 | `harmonization/src/harm_apply.py` | Apply saved harmonization model to test CSV |
| 6 | `data_utils/data_selection/src/merge_files.py` | Build train/test CSVs for each SPARE task |
| 7 | `spare_scores/src/spare_train.py` | Train linear SVM (sklearn) with n-fold CV; saves `.joblib` model |
| 8 | `spare_scores/src/spare_apply.py` | Apply saved SPARE model; produce score CSV |

Harmonization and combat steps are **skipped** if the corresponding `tasks/<task>/` folder does not exist in the input directory. SPARE tasks always run unless `SPARE_TASKS = []` is set in `s3_run_pipelines.py`.

NiChart SPARE variants (`nichart_spare_train.py` / `nichart_spare_apply.py`) are an alternative to the sklearn SPARE pipeline.

### `merge_files.py` — task descriptor format

`merge_files.py` is the shared data-assembly utility. Each task folder contains one or more JSON descriptor files:

- `data_desc.json` — used by QC pipeline
- `training_data_desc.json` / `testing_data_desc.json` — used by harmonization and SPARE pipelines

Descriptor keys: `sample` (MRID list), `data` (column specs referencing `dict_var_groups.json` + `dict_data_files.json`, optional `filters`, `mappings`, `additional_vars`), `output` (path + filename). Column specs use the form `"file_key.group_name"` or `"file_key.[Col1, Col2]"`.

### `harm_train.py` — harmonization model

- CSV column layout: col 1 = MRID, col 2 = batch variable, cols 3–5 = covariates (col 3 gets a natural cubic spline for age), col 6+ = data variables.
- Fits OLS per variable: `y ~ C(batch) + cr(age, df=5) + cov2 + cov3`. Removes batch effect: `y_harm = y - X_batch @ beta_batch`.
- Model saved as a plain dict via `joblib.dump` (not a patsy DesignInfo object) so it can be reloaded cross-environment.

### `spare_train.py` — SPARE scoring model

- CSV column layout: col 1 = MRID, col 2 = target, col 3+ = features.
- Auto-detects classification (CL) vs regression (RG) from the target column.
- Trains `Pipeline(StandardScaler, LinearSVC/LinearSVR)` with 10-fold CV. Final model is fit on all training data and saved as `.joblib`.

### Logging

All pipeline scripts use `utils/logger.py:setup_logger()`. Console output is INFO by default; `-v` enables DEBUG. When a `log_dir` is provided, a timestamped `.log` file captures full DEBUG output.

### Idempotency

Every individual script checks whether its output files already exist and skips re-computation if so. Re-running the pipeline after partial failure is safe.
