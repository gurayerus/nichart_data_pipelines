"""
Create an anonymized test dataset.

Steps:
  1. Anonymize    — anonymize the complete data CSV; save
                    {stem}_full_anon.csv and {stem}_mappings.json to
                    input/samples/sample_anon_data/ (private — contain original IDs)
  2. Sample       — if --sample_csv is given, translate original MRIDs to anon
                    MRIDs and filter the full anon CSV to the sample; save as
                    out_dir/data/{stem}_anon.csv.  If no sample_csv, the full
                    anonymized CSV is copied directly to out_dir/data/.
  3. Dictionaries — copy input/dictionaries/ → out_dir/dictionaries/ verbatim
  4. Lists        — copy input/tasks/list_*.csv to out_dir/tasks/ with MRID
                    and any other mapped columns (e.g. batch) anonymized;
                    rows whose MRID is not in the mapping are dropped

Default paths (relative to this script):
  --data_csv   ../../../input/data/istaging_3_0_harmonized.csv
  --sample_csv ../../../input/samples/test-s1/test-s1.csv
  --input_dir  ../../../input
  --out_dir    ../../../input_anon
  --out_stem   istaging_test

Usage:
  python create_test_dataset.py
  python create_test_dataset.py --out_dir <path> --out_stem <stem>
"""

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).parent


# ── helpers ───────────────────────────────────────────────────────────────────

def load_mrid_set(sample_csv: Path) -> set[str]:
    with sample_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        mrids  = {row[0].strip() for row in reader if row and row[0].strip()}
    print(f"  Sample list : {sample_csv.name}  ({len(mrids)} MRIDs)"
          + (f"  [key: '{header[0]}']" if header else ""))
    return mrids


# ── steps ─────────────────────────────────────────────────────────────────────

def step1_anonymize(
    data_csv: Path,
    stem: str,
    anon_dir: Path,
) -> Path:
    """Anonymize the complete data CSV; save full anon CSV and mappings to anon_dir.

    Outputs (both private):
      {stem}_full_anon.csv   — full anonymized dataset
      {stem}_mappings.json   — orig → anon ID mapping
    Skips if both already exist.  Returns path to mappings file.
    """
    from anon_sample import anonymize

    full_anon_dst = anon_dir / f"{stem}_full_anon.csv"
    mappings_dst  = anon_dir / f"{stem}_mappings.json"

    if full_anon_dst.exists() and mappings_dst.exists():
        print(f"  Skipping: {full_anon_dst.name} and {mappings_dst.name} already exist.")
        return mappings_dst

    if not data_csv.exists():
        sys.exit(f"File not found: {data_csv}")

    anon_dir.mkdir(parents=True, exist_ok=True)

    with data_csv.open(newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows       = list(reader)

    print(f"  Loaded {len(rows)} rows from {data_csv.name}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{stem}.csv", newline="", encoding="utf-8", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        csv.DictWriter(tmp, fieldnames=fieldnames).writeheader()
        csv.DictWriter(tmp, fieldnames=fieldnames).writerows(rows)

    try:
        anonymize(tmp_path)
        anon_src     = tmp_path.with_name(tmp_path.stem + "_anon.csv")
        mappings_src = tmp_path.with_name(tmp_path.stem + "_mappings.json")

        for src, dst in ((anon_src, full_anon_dst), (mappings_src, mappings_dst)):
            if dst.exists():
                print(f"  Skipping: {dst.name} already exists.")
            else:
                shutil.move(str(src), dst)
                print(f"  Written {dst.name}  →  {dst.parent}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return mappings_dst

def step2_anonymize_samples(input_dir: Path, out_dir: Path, mappings_path: Path):
    """Copy input/tasks/list_*.csv to out_dir/tasks/ with mapped columns anonymized.

    All columns present in the mappings JSON (e.g. MRID, batch) are replaced with
    their anonymized equivalents.  Rows whose MRID is not in the mapping are dropped.
    The relative subdirectory structure under tasks/ is preserved.
    """
    src_samples = input_dir / "samples"
    dst_samples = out_dir   / "samples"

    if not src_samples.exists():
        print(f"  Warning: {src_samples} not found, skipping.", file=sys.stderr)
        return

    with mappings_path.open(encoding="utf-8") as f:
        mappings: dict = json.load(f)

    # col_name → {orig_val: anon_val}
    col_maps: dict[str, dict] = {col: m for col, m in mappings.items() if isinstance(m, dict)}
    mrid_map = col_maps.get("MRID", {})
    print(f"  Column maps available: {sorted(col_maps)}")

    list_files = sorted(src_samples.rglob("*_list.csv"))
    if not list_files:
        print("  No list_*.csv files found under input/samples/.")
        return

    for src_file in list_files:
        rel      = src_file.relative_to(src_samples)
        dst_file = dst_samples / rel
        if dst_file.exists():
            print(f"  Skipping: {rel} already exists.")
            continue

        dst_file.parent.mkdir(parents=True, exist_ok=True)

        with src_file.open(newline="", encoding="utf-8") as f:
            reader     = csv.DictReader(f)
            fieldnames = reader.fieldnames
            mrid_col   = fieldnames[0]
            rows       = list(reader)

        # Columns in this file (beyond MRID) that have a mapping entry
        extra_cols = [c for c in fieldnames if c != mrid_col and c in col_maps]

        kept, dropped = [], 0
        for row in rows:
            anon_mrid = mrid_map.get(row[mrid_col])
            if anon_mrid is None:
                dropped += 1
                continue
            new_row = dict(row)
            new_row[mrid_col] = anon_mrid
            for col in extra_cols:
                val = row[col]
                new_row[col] = col_maps[col].get(val, val)  # leave unmapped values as-is
            kept.append(new_row)

        with dst_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

        extra_note = f"  [also mapped: {extra_cols}]" if extra_cols else ""
        print(f"  {rel}: {len(kept)} rows kept"
              + (f", {dropped} dropped (MRID not in mapping)" if dropped else "")
              + extra_note)

def step3_sample(
    stem: str,
    anon_dir: Path,
    sample_csv: Path | None,
    mappings_path: Path,
    out_data_dir: Path,
) -> None:
    """Filter the full anon CSV to sample MRIDs and save to out_data_dir.

    If sample_csv is None, copies the full anonymized CSV as-is.
    Skips if output file already exists.
    """
    full_anon_src = anon_dir / f"{stem}_full_anon.csv"
    out_dst       = out_data_dir        / f"{stem}_anon.csv"

    if out_dst.exists():
        print(f"  Skipping: {out_dst.name} already exists.")
        return

    out_data_dir.mkdir(parents=True, exist_ok=True)

    with full_anon_src.open(newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames
        anon_rows  = list(reader)

    if sample_csv is None:
        rows_out = anon_rows
        print(f"  No sample CSV — using all {len(rows_out)} rows")
    else:

        if sample_csv is not None and not sample_csv.exists():
            sys.exit(f"File not found: {sample_csv}")

        with mappings_path.open(encoding="utf-8") as f:
            mrid_map: dict = json.load(f).get("MRID", {})  # orig → anon

        orig_mrids = load_mrid_set(sample_csv)
        anon_mrids = {mrid_map[m] for m in orig_mrids if m in mrid_map}
        missed     = orig_mrids - set(mrid_map)
        if missed:
            print(f"  Warning: {len(missed)} sample MRIDs not found in mapping.",
                  file=sys.stderr)

        mrid_col = fieldnames[0]
        rows_out = [r for r in anon_rows if r[mrid_col] in anon_mrids]
        print(f"  Sampled {len(rows_out)} / {len(anon_rows)} rows")

    with out_dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"  Written {out_dst.name}  →  {out_dst.parent}")

def step3_copy_data(input_dir: Path, out_dir: Path):
    """Copy anonymized input/data/ → out_dir/data/ verbatim."""
    src = input_dir / "data"
    dst = out_dir   / "data"

    if not src.exists():
        print(f"  Warning: {src} not found, skipping.", file=sys.stderr)
        return

    if dst.exists():
        print(f"  Skipping: {dst} already exists.")
        return

    shutil.copytree(src, dst)
    n = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"  Copied  {src.name}/  ({n} files)  →  {dst}")

def step4_copy_samples(input_dir: Path, out_dir: Path):
    """Copy anonymized input/samples/ → out_dir/samples/ verbatim."""
    src = input_dir / "samples"
    dst = out_dir   / "samples"

    if not src.exists():
        print(f"  Warning: {src} not found, skipping.", file=sys.stderr)
        return

    if dst.exists():
        print(f"  Skipping: {dst} already exists.")
        return

    shutil.copytree(src, dst)
    n = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"  Copied  {src.name}/  ({n} files)  →  {dst}")



def step5_copy_dicts(input_dir: Path, out_dir: Path):
    """Copy input/dictionaries/ → out_dir/dictionaries/ verbatim."""
    src = input_dir / "dictionaries"
    dst = out_dir   / "dictionaries"

    if not src.exists():
        print(f"  Warning: {src} not found, skipping.", file=sys.stderr)
        return

    if dst.exists():
        print(f"  Skipping: {dst} already exists.")
        return

    shutil.copytree(src, dst)
    n = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"  Copied  {src.name}/  ({n} files)  →  {dst}")

def step6_copy_tasks(input_dir: Path, out_dir: Path):
    """Copy input/tasks/ → out_dir/tasks/ verbatim."""
    src = input_dir / "tasks"
    dst = out_dir   / "tasks"

    if not src.exists():
        print(f"  Warning: {src} not found, skipping.", file=sys.stderr)
        return

    if dst.exists():
        print(f"  Skipping: {dst} already exists.")
        return

    shutil.copytree(src, dst)
    n = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"  Copied  {src.name}/  ({n} files)  →  {dst}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build anonymized test dataset.")
    parser.add_argument("--data_csv",   required=True,
                        help="full iSTAGING input CSV")
    parser.add_argument("--sample_csv", default=None,
                        help="CSV whose first column lists the MRIDs to keep "
                             "(omit to anonymize the full data CSV)")
    parser.add_argument("--input_dir",  required=True,
                        help="private input directory (source of dictionaries/ and tasks/)")
    parser.add_argument("--out_dir",    required=True,
                        help="anonymized output directory root")
    parser.add_argument("--out_stem",   required=True,
                        help="filename stem for the output data file")
    args = parser.parse_args()

    data_csv   = Path(args.data_csv)
    sample_csv = Path(args.sample_csv) if args.sample_csv else None
    input_dir  = Path(args.input_dir)
    out_dir    = Path(args.out_dir)
    stem       = args.out_stem

    out_anon_dir = input_dir / "data_anon"

    bar = "─" * 56
    print(f"\n{bar}\n  Step 1 — Anonymize data\n{bar}")
    mappings_path = step1_anonymize(data_csv, stem, out_anon_dir)

    print(f"\n{bar}\n  Step 2 — Anonymize lists\n{bar}")
    step2_anonymize_samples(input_dir, out_anon_dir, mappings_path)

    print(f"\n{bar}\n  Step 3 — Select sample\n{bar}")
    step3_sample(stem, out_anon_dir, sample_csv, mappings_path, out_dir / "data")

    print(f"\n{bar}\n  Step 4 — Copy anonymized samples\n{bar}")
    step4_copy_samples(out_anon_dir, out_dir)

    print(f"\n{bar}\n  Step 5— Copy dictionaries\n{bar}")
    step5_copy_dicts(input_dir, out_dir)

    print(f"\n{bar}\n  Step 6 — Copy tasks\n{bar}")
    step6_copy_tasks(input_dir, out_dir)


if __name__ == "__main__":
    main()
