"""
Select a sample from an anonymized dataset.

Reads the sample list from in_dir/samples/<in_sample_csv>, then:
  1. data/         — filter every CSV in in_dir/data/ to sample MRIDs
                     → out_dir/data/
  2. samples/      — filter every CSV under in_dir/samples/ to sample MRIDs
                     → out_dir/samples/  (directory structure preserved)
  3. dictionaries/ — copy in_dir/dictionaries/ → out_dir/dictionaries/
  4. tasks/        — copy in_dir/tasks/        → out_dir/tasks/

Usage:
  python run_select_sample.py \\
      --in_dir        data_anon_full \\
      --in_sample_csv test-s1/test-s1_list.csv \\
      --out_dir       test/data_anon
  python run_select_sample.py ... -v
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

_UTILS = Path(__file__).resolve().parent / "utils"
sys.path.insert(0, str(_UTILS))
from logger import setup_logger  # type: ignore[import]

TOTAL_STEPS = 4
_log = logging.getLogger("run_select_sample")


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _log.info(f"    Skipping: {path.name} already exists.")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _log.info(f"    Written : {path}")


def _filter_dir(src: Path, dst: Path, sample_mrids: set[str], label: str) -> None:
    if not src.exists():
        _log.warning(f"  {label}: source not found ({src}) — skipping")
        return
    csvs = sorted(src.rglob("*.csv"))
    if not csvs:
        _log.info(f"  {label}: no CSV files found")
        return
    for csv_path in csvs:
        rel = csv_path.relative_to(src)
        _log.info(f"  » {rel}")
        fieldnames, all_rows = _read_csv(csv_path)
        kept   = [r for r in all_rows if r.get("MRID", "").strip() in sample_mrids]
        missed = len(all_rows) - len(kept)
        _log.info(f"    {len(kept)} / {len(all_rows)} rows selected"
                  + (f"  ({missed} not in sample)" if missed else ""))
        _write_csv(dst / rel, fieldnames, kept)


def _copy_dir(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        _log.warning(f"  {label}: source not found ({src}) — skipping")
        return
    if dst.exists():
        _log.info(f"  {label}: Skipping — {dst} already exists.")
        return
    shutil.copytree(src, dst)
    n = sum(1 for p in dst.rglob("*") if p.is_file())
    _log.info(f"  {label}: copied {n} file(s)  →  {dst}")


def main():
    parser = argparse.ArgumentParser(
        description="Select a sample from an anonymized dataset."
    )
    parser.add_argument("--in_dir",        required=True,
                        help="input root containing data/, samples/, "
                             "dictionaries/, tasks/ subfolders")
    parser.add_argument("--in_sample_csv", required=True,
                        help="sample list CSV relative to in_dir/samples/ "
                             "(MRID in first column)")
    parser.add_argument("--out_dir",       required=True,
                        help="output root; same subfolder structure will be created")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show debug output on the console")
    parser.add_argument("--log_dir",       default=None,
                        help="directory for log file")
    args = parser.parse_args()

    setup_logger("run_select_sample", verbose=args.verbose,
                 log_dir=Path(args.log_dir) if args.log_dir else None)
    _log.debug("Command: " + " ".join(sys.argv))

    in_dir     = Path(args.in_dir)
    out_dir    = Path(args.out_dir)
    sample_csv = in_dir / "samples" / args.in_sample_csv

    # ── read sample MRIDs ─────────────────────────────────────────────────────
    if not sample_csv.exists():
        sys.exit(f"Error: sample list not found: {sample_csv}")
    with sample_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        mrid_col = reader.fieldnames[0]
        sample_mrids = {row[mrid_col].strip() for row in reader}
    _log.info(f"  Sample : {sample_csv}  ({len(sample_mrids)} MRIDs)")

    # ── Step 1: filter data/ ──────────────────────────────────────────────────
    _log.info(f"\n[STEP 1/{TOTAL_STEPS}]  data — filter CSVs to sample")
    _filter_dir(in_dir / "data", out_dir / "data", sample_mrids, "data")
    _log.info(f"\n  SUCCESS: data")

    # ── Step 2: filter samples/ ───────────────────────────────────────────────
    _log.info(f"\n[STEP 2/{TOTAL_STEPS}]  samples — filter CSVs to sample")
    _filter_dir(in_dir / "samples", out_dir / "samples", sample_mrids, "samples")
    _log.info(f"\n  SUCCESS: samples")

    # ── Step 3: copy dictionaries/ ────────────────────────────────────────────
    _log.info(f"\n[STEP 3/{TOTAL_STEPS}]  dictionaries — copy verbatim")
    _copy_dir(in_dir / "dictionaries", out_dir / "dictionaries", "dictionaries")
    _log.info(f"\n  SUCCESS: dictionaries")

    # ── Step 4: copy tasks/ ───────────────────────────────────────────────────
    _log.info(f"\n[STEP 4/{TOTAL_STEPS}]  tasks — copy verbatim")
    _copy_dir(in_dir / "tasks", out_dir / "tasks", "tasks")
    _log.info(f"\n  SUCCESS: tasks")

    _log.info("\nSample selection complete.")


if __name__ == "__main__":
    main()
