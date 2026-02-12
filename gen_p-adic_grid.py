#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_set(p: int, n: int) -> list[int]:
    # A = { i * p**j : 1 <= i,j <= n }
    # Use a Python set to remove duplicates, then sort for deterministic output.
    values: set[int] = set()
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            values.add(i * (p**j))
    return sorted(values)


def ensure_csv_header(path: Path, header: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
        return

    # If the file already exists, ensure the header matches.
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        existing = next(r, None)
    if existing != header:
        raise SystemExit(
            f"CSV header mismatch in {path}.\n"
            f"Expected: {header}\n"
            f"Found:    {existing}\n"
            "Use a different --output file or replace the existing file."
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "For each k=1..n, build A_k = { i * p**j : 1 <= i,j <= k } and record |A|, |A+A|, |A*A| using OOKAMI."
        )
    )
    ap.add_argument("--p", type=int, required=True, help="Prime p (base of the powers).")
    ap.add_argument("--n", type=int, required=True, help="Positive integer n (compute k=1..n).")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: data/p-adic_grid_data_<p>_<n>.csv).",
    )
    args = ap.parse_args()

    if args.p <= 1:
        raise SystemExit("--p must be > 1")
    if args.n <= 0:
        raise SystemExit("--n must be > 0")

    try:
        from ookami import CombSet  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Could not import OOKAMI (module 'ookami').\n"
            "Install it into this environment, then re-run.\n"
            f"Details: {e}"
        )

    header = ["set_cardinality", "add_ds_card", "mult_ds_card", "count"]
    out_path = args.output if args.output is not None else Path(f"data/p-adic_grid_data_{args.p}_{args.n}.csv")
    ensure_csv_header(out_path, header)

    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k in range(1, args.n + 1):
            s = build_set(args.p, k)
            A = CombSet(s)
            w.writerow([int(A.cardinality), int(A.ads_cardinality), int(A.mds_cardinality), 1])


if __name__ == "__main__":
    main()
