#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from itertools import combinations
from pathlib import Path


def generate_rows(n: int) -> list[tuple[int, int, int, int]]:
    try:
        from ookami import CombSet  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Could not import OOKAMI (module 'ookami').\n"
            "Install it in your current Python environment, then re-run.\n"
            f"Details: {e}"
        )

    universe = list(range(1, n + 1))
    min_size = n - 5

    counts: Counter[tuple[int, int, int]] = Counter()

    for size in range(min_size, n + 1):
        for subset in combinations(universe, size):
            comb_set = CombSet(list(subset))
            key = (
                int(comb_set.cardinality),
                int(comb_set.ads_cardinality),
                int(comb_set.mds_cardinality),
            )
            counts[key] += 1

    return [
        (set_cardinality, add_ds_card, mult_ds_card, count)
        for (set_cardinality, add_ds_card, mult_ds_card), count in sorted(counts.items())
    ]


def write_csv(path: Path, rows: list[tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set_cardinality", "add_ds_card", "mult_ds_card", "count"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute aggregated (|A|, |A+A|, |A*A|, count) for all subsets A of [n] "
            "with sizes in [n-5, n], using OOKAMI."
        )
    )
    parser.add_argument("n", type=int, help="n for [n] = {1, ..., n}; must be >= 7")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/progressions_n<N>.csv)",
    )
    args = parser.parse_args()

    if args.n < 7:
        raise SystemExit("n must be >= 7")

    output_path = args.output if args.output is not None else Path(f"data/progressions_n{args.n}.csv")

    rows = generate_rows(args.n)
    write_csv(output_path, rows)


if __name__ == "__main__":
    main()
