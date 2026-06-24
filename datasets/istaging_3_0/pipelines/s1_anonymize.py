"""
Anonymize an iSTAGING dataset.

Expects in_dir to contain the subfolders:
  data/          — CSV files; in_csv is the reference for building the mapping
  samples/       — CSV files to anonymize (directory structure preserved)
  dictionaries/  — copied verbatim to out_dir/dictionaries/
  tasks/         — copied verbatim to out_dir/tasks/

Outputs mirror the same structure under out_dir:
  out_dir/data/           — anonymized CSV files
  out_dir/samples/        — anonymized sample CSV files
  out_dir/dictionaries/   — verbatim copy
  out_dir/tasks/          — verbatim copy
  out_dir/anon_mapping/   — <in_csv_stem>_mappings.json  (keep private)

Usage:
  python run_anonymize.py \\
      --in_dir  data_private \\
      --in_csv  istaging_3_0_harmonized.csv \\
      --out_dir data_anon_full
  python run_anonymize.py ... -v
"""

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

_ANON_SRC = Path(__file__).resolve().parent / "data_utils/data_init/src"
sys.path.insert(0, str(_ANON_SRC))
from anon_sample import build_mappings, COLUMNS  # type: ignore[import]

_UTILS = Path(__file__).resolve().parent / "utils"
sys.path.insert(0, str(_UTILS))
from logger import setup_logger  # type: ignore[import]

TOTAL_STEPS = 4
_log = logging.getLogger("run_anonymize")


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


def _apply(rows: list[dict], mappings: dict, drop_missing_mrid: bool = True) -> list[dict]:
    """Apply anonymization mappings to rows."""
    mrid_map = mappings.get("MRID", {})
    kept, dropped = [], 0
    for row in rows:
        if drop_missing_mrid and row.get("MRID", "") not in mrid_map:
            dropped += 1
            continue
        new_row = dict(row)
        for col, col_map in mappings.items():
            if col in row and row[col] in col_map:
                new_row[col] = col_map[row[col]]
        kept.append(new_row)
    if dropped:
        _log.debug(f"      {dropped} rows dropped (MRID not in mapping)")
    return kept


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
    parser = argparse.ArgumentParser(description="Anonymize iSTAGING dataset.")
    parser.add_argument("--in_dir",  required=True,
                        help="input root containing data/, samples/, "
                             "dictionaries/, tasks/ subfolders")
    parser.add_argument("--in_csv",  required=True,
                        help="filename of the main CSV inside in_dir/data/ "
                             "used to build the anonymization mapping")
    parser.add_argument("--out_dir", required=True,
                        help="output root; mirrors the same subfolder structure")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show debug output on the console")
    parser.add_argument("--log_dir",  default=None,
                        help="directory for log file")
    args = parser.parse_args()

    setup_logger("run_anonymize", verbose=args.verbose,
                 log_dir=Path(args.log_dir) if args.log_dir else None)
    _log.debug("Command: " + " ".join(sys.argv))

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    in_csv  = in_dir / "data" / args.in_csv

    # ── Step 1: build mapping + anonymize data/ ───────────────────────────────
    _log.info(f"\n[STEP 1/{TOTAL_STEPS}]  data — anonymize CSVs in {in_dir / 'data'}")

    if not in_csv.exists():
        sys.exit(f"Error: reference CSV not found: {in_csv}")

    fieldnames, rows = _read_csv(in_csv)
    active_cols = {c: p for c, p in COLUMNS.items() if c in fieldnames}
    if not active_cols:
        sys.exit(f"Error: none of the anonymization columns found in {in_csv.name}")

    mappings = build_mappings(rows, active_cols)
    _log.info("  Columns mapped: "
              + ", ".join(f"{c} ({len(m)} values)" for c, m in mappings.items()))

    # Save mapping JSON to out_dir/anon_mapping/
    mapping_path = out_dir / "anon_mapping" / f"{in_csv.stem}_mappings.json"
    if mapping_path.exists():
        _log.info(f"  Skipping mapping file: {mapping_path.name} already exists.")
    else:
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.write_text(json.dumps(mappings, indent=2), encoding="utf-8")
        _log.info(f"  Mapping saved: {mapping_path}")

    # Anonymize all CSVs in in_dir/data/
    data_csvs = sorted((in_dir / "data").glob("*.csv"))
    if not data_csvs:
        _log.warning(f"  No CSV files found in {in_dir / 'data'}")
    for csv_path in data_csvs:
        _log.info(f"  » {csv_path.name}")
        fnames, frows = _read_csv(csv_path)
        out_rows = _apply(frows, mappings, drop_missing_mrid=False)
        _write_csv(out_dir / "data" / csv_path.name, fnames, out_rows)

    _log.info(f"\n  SUCCESS: {len(data_csvs)} data CSV(s) anonymized")

    # ── Step 2: anonymize samples/ ────────────────────────────────────────────
    _log.info(f"\n[STEP 2/{TOTAL_STEPS}]  samples — anonymize CSVs in {in_dir / 'samples'}")

    in_samples = in_dir / "samples"
    if not in_samples.exists():
        _log.warning(f"  samples/ not found — skipping")
    else:
        sample_csvs = sorted(in_samples.rglob("*.csv"))
        if not sample_csvs:
            _log.info("  No CSV files found under samples/")
        for csv_path in sample_csvs:
            rel = csv_path.relative_to(in_samples)
            _log.info(f"  » {rel}")
            fnames, frows = _read_csv(csv_path)
            out_rows = _apply(frows, mappings, drop_missing_mrid=True)
            _write_csv(out_dir / "samples" / rel, fnames, out_rows)
            _log.info(f"    {len(out_rows)} rows written")
        _log.info(f"\n  SUCCESS: {len(sample_csvs)} sample CSV(s) anonymized")

    # ── Step 3: copy dictionaries/ ────────────────────────────────────────────
    _log.info(f"\n[STEP 3/{TOTAL_STEPS}]  dictionaries — copy to {out_dir / 'dictionaries'}")
    _copy_dir(in_dir / "dictionaries", out_dir / "dictionaries", "dictionaries")
    _log.info(f"\n  SUCCESS: dictionaries")

    # ── Step 4: copy tasks/ ───────────────────────────────────────────────────
    _log.info(f"\n[STEP 4/{TOTAL_STEPS}]  tasks — copy to {out_dir / 'tasks'}")
    _copy_dir(in_dir / "tasks", out_dir / "tasks", "tasks")
    _log.info(f"\n  SUCCESS: tasks")

    _log.info("\nAnonymization complete.")


if __name__ == "__main__":
    main()
