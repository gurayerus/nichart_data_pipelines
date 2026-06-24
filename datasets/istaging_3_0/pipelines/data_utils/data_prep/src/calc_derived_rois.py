"""
Calculate derived H_DLMUSE ROI values by summing their component single ROIs.

Reads:
  istag_hdlmuse.csv        — MRID + single H_DLMUSE_{index} columns (index < 255)
  muse_mapping_derived.csv — derived ROI definitions
      col 0: derived ROI index
      col 1: name (ignored)
      col 2+: component ROI indices (empty strings = padding)

For each derived ROI, value = sum of its component H_DLMUSE_{index} columns.
Derived ROIs where any component is absent from the input are skipped entirely.

Output:
  istag_hdlmuse_derived.csv — MRID + H_DLMUSE_{derived_index} columns
"""

import argparse
import csv
from pathlib import Path

THIS_DIR = Path(__file__).parent

DEFAULTS = {
    "input_csv":  THIS_DIR / "../../../output/data_prep/istag_hdlmuse.csv",
    "mapping":    THIS_DIR / "../../../input/dictionaries/muse_mapping_derived.csv",
    "output_csv": THIS_DIR / "../../../output/data_prep/istag_hdlmuse_all.csv",
}

KEY_COL = "MRID"
COL_PREFIX = "H_DLMUSE_"


def load_mapping(path: Path) -> dict[int, list[int]]:
    """Return {derived_idx: [component_idx, ...]} with padding stripped."""
    mapping = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            derived_idx = int(row[0])
            components = [int(x) for x in row[2:] if x.strip()]
            if components:
                mapping[derived_idx] = components
    return mapping


def calc_derived(input_csv: Path, mapping_csv: Path, output_csv: Path):
    if output_csv.exists():
        print(f"Skipping: {output_csv} already exists.")
        return

    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    available = {
        int(c[len(COL_PREFIX):])
        for c in fieldnames
        if c.startswith(COL_PREFIX)
    }

    mapping = load_mapping(mapping_csv)

    valid: dict[int, list[int]] = {}
    skipped: list[tuple[int, list[int]]] = []
    for derived_idx, components in mapping.items():
        missing = [c for c in components if c not in available]
        if missing:
            skipped.append((derived_idx, missing))
        else:
            valid[derived_idx] = components

    out_indices = list(valid.keys())  # preserve mapping file order
    out_fieldnames = [KEY_COL] + [f"{COL_PREFIX}{idx}" for idx in out_indices]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {KEY_COL: row[KEY_COL]}
            for derived_idx in out_indices:
                out_row[f"{COL_PREFIX}{derived_idx}"] = sum(
                    float(row[f"{COL_PREFIX}{c}"]) for c in valid[derived_idx]
                )
            writer.writerow(out_row)

    print(f"Input : {input_csv}  ({len(rows)} rows, {len(available)} single ROIs)")
    print(f"Output: {output_csv}  ({len(out_indices)} derived ROIs)")
    if skipped:
        print(f"Skipped {len(skipped)} derived ROI(s) — missing components:")
        for derived_idx, missing in skipped:
            print(f"  H_DLMUSE_{derived_idx}: missing {missing}")


def main():
    parser = argparse.ArgumentParser(description="Calculate derived H_DLMUSE ROI values.")
    parser.add_argument("-i", "--input_csv",  default=DEFAULTS["input_csv"],  help="path to istag_hdlmuse.csv")
    parser.add_argument("-m", "--mapping",    default=DEFAULTS["mapping"],    help="path to muse_mapping_derived.csv")
    parser.add_argument("-o", "--output_csv", default=DEFAULTS["output_csv"], help="path for output CSV")
    args = parser.parse_args()

    calc_derived(
        input_csv=Path(args.input_csv),
        mapping_csv=Path(args.mapping),
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
