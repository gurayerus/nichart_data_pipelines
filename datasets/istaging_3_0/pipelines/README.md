# pipelines

Data processing pipelines for the iSTAGING 3.0 dataset.

## Directory layout

```
pipelines/
  run_init.py          — Step 0: anonymize + sample the raw input CSV → input_anon/
  run_pipelines.py     — Steps 1–8: full data processing pipeline
  run_qc.py            — QC / distribution verification

  data_utils/          — shared data manipulation utilities
    data_init/src/
      create_sample.py     anonymize full dataset and optionally draw a sample
    data_prep/src/
      split_input.py       split anonymized CSV into themed sub-tables
      calc_derived_rois.py compute derived H_DLMUSE ROI columns
    data_selection/src/
      merge_files.py    filter + reshape data into train/test CSVs per task
    data_qc/src/
      create_verification_data.py  merge columns for distribution checks
      plot_distributions.py        generate distribution plots

  harmonization/src/   — univariate batch harmonization pipeline
    harm_train.py          fit OLS batch-effect models on training set
    harm_test.py           apply models to produce harmonized values

  spare_scores/src/    — SPARE biomarker scoring pipeline
    spare_train.py         train linear SVM model (n-fold CV)
    spare_test.py          apply model for inference

  ref_centiles/        — reference centile computation (in progress)
```

## Typical workflow

```
run_init.py  →  run_pipelines.py  →  run_qc.py
```

### Step 0 — `run_init.py`

Anonymizes the raw iSTAGING CSV and optionally draws a sample.
Produces the `input_anon/` directory consumed by all downstream steps.

```bash
cd datasets/istaging_3_0/pipelines
python run_init.py
# override defaults:
python run_init.py --data_csv <full_csv> --sample_csv <mrid_list> --out_stem istaging_test
```

Calls `data_utils/data_init/src/create_sample.py` with all paths resolved here.
Omit `--sample_csv` to anonymize the full dataset without sub-sampling.

### Steps 1–8 — `run_pipelines.py`

Runs the complete processing chain end-to-end, reading from `input_anon/` and
writing all outputs under `output/`.

```bash
cd datasets/istaging_3_0/pipelines
python run_pipelines.py
```

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_utils/data_prep` / `split_input.py` | Split anonymized CSV into themed sub-tables |
| 2 | `data_utils/data_prep` / `calc_derived_rois.py` | Compute derived H_DLMUSE ROI columns |
| 3 | `data_utils/data_selection` / `merge_files.py` | Build harmonization train + test CSVs |
| 4 | `harmonization` / `harm_train.py` | Fit batch-effect harmonization models |
| 5 | `harmonization` / `harm_test.py` | Apply models → harmonized values |
| 6 | `data_utils/data_selection` / `merge_files.py` | Build SPARE train + test CSVs per task |
| 7 | `spare_scores` / `spare_train.py` | Train SPARE linear SVM models |
| 8 | `spare_scores` / `spare_test.py` | Apply SPARE models → scores |

Final outputs are symlinked into `output/final/data/` and `output/final/models/`.

### QC — `run_qc.py`

Merges columns from `output/final/data/` and generates distribution plots per verification task.

```bash
cd datasets/istaging_3_0/pipelines
python run_qc.py
python run_qc.py --tasks dlmuse601_distributions
```

## Output layout

```
output/
  final/
    data/            — symlinks to key CSVs (harmonized data, SPARE scores)
    models/          — symlinks to trained model .joblib files
  intermediate/
    harmonization/   — harm_train / harm_test working files
    spare_scores/
      spare-ad-raw/  — per-task working files
      spare-ad-h/
      spare-ad-h2/
  qc/                — distribution plots and merged verification CSVs
```
