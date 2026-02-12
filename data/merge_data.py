#!/usr/bin/env python3
import csv
import glob
import os
import sys
from collections import defaultdict

def read_base_pairs(path: str) -> defaultdict[tuple[int, int, int], int]:
    merged = defaultdict(int)
    if not os.path.exists(path):
        return merged

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"set_cardinality", "add_ds_card", "mult_ds_card", "count"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} missing required columns: {required}")

        for row in reader:
            key = (int(row["set_cardinality"]), int(row["add_ds_card"]), int(row["mult_ds_card"]))
            merged[key] += int(row["count"])
    return merged

def merge_spp_files(merged: defaultdict[tuple[int, int, int], int], pattern: str = "spp_k*.csv") -> None:
    for fp in sorted(glob.glob(pattern)):
        with open(fp, newline="") as f:
            reader = csv.DictReader(f)
            required = {"set_cardinality", "add_ds_card", "mult_ds_card", "count"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{fp} missing required columns: {required}")

            for row in reader:
                key = (int(row["set_cardinality"]), int(row["add_ds_card"]), int(row["mult_ds_card"]))
                merged[key] += int(row["count"])

def write_merged_csv(path: str, merged: dict[tuple[int, int, int], int]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["set_cardinality", "add_ds_card", "mult_ds_card", "count"])
        for (sc, ad, md), c in sorted(merged.items()):
            writer.writerow([sc, ad, md, c])

def main() -> int:
    base_file = "data.csv"
    out_file = "new-data.csv"

    merged = read_base_pairs(base_file)
    merge_spp_files(merged, "p-adic*.csv")
    write_merged_csv(out_file, merged)

    print(f"Wrote {out_file} with {len(merged)} unique points.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

