"""
Anonymize identifying columns in a CSV file.

Columns anonymized (if present):
  MRID, PTID, Study, Group, SITE, Phase, Study_full,
  original_batch, batch, MRID_2_0

Outputs:
  <stem>_anon.csv       — anonymized CSV
  <stem>_mappings.json  — dict of original → anonymous values per column
"""

import csv
import json
import sys
from pathlib import Path

COLUMNS = {
    "MRID":           "scan",
    "PTID":           "subj",
    "Study":          "study",
    "Group":          "group",
    "SITE":           "site",
    "Phase":          "phase",
    "Study_full":     "studyfull",
    "original_batch": "origbatch",
    "batch":          "batch",
    "MRID_2_0":       "scan2",
}


def build_mappings(rows, columns):
    mappings = {col: {} for col in columns}
    for row in rows:
        for col, prefix in columns.items():
            val = row.get(col)
            if val is not None and val not in mappings[col]:
                idx = len(mappings[col]) + 1
                mappings[col][val] = f"{prefix}{idx}"
    return mappings


def anonymize(input_path: Path):
    out_csv = input_path.with_name(input_path.stem + "_anon.csv")
    if out_csv.exists():
        print(f"Skipping: {out_csv} already exists.")
        return

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    missing = [c for c in COLUMNS if c not in fieldnames]
    if missing:
        print(f"Warning: columns not found in CSV (skipped): {missing}", file=sys.stderr)

    active_columns = {c: p for c, p in COLUMNS.items() if c in fieldnames}
    mappings = build_mappings(rows, active_columns)

    anon_rows = []
    for row in rows:
        new_row = dict(row)
        for col, mapping in mappings.items():
            if row[col] is not None:
                new_row[col] = mapping[row[col]]
        anon_rows.append(new_row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(anon_rows)

    out_json = input_path.with_name(input_path.stem + "_mappings.json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)

    print(f"Anonymized CSV : {out_csv}")
    print(f"Mappings JSON  : {out_json}")
    for col, m in mappings.items():
        print(f"  {col}: {len(m)} unique values")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("istaging_3_0_test_v3.csv")
    if not path.exists():
        sys.exit(f"File not found: {path}")
    anonymize(path)
