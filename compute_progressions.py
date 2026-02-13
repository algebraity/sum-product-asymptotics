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


def compute_data(n: int, path: str = None) -> None:
    if n < 7:
        raise SystemExit("n must be >= 7")

    output_path = Path(path) if path is not None else Path(f"data/progressions_n{n}.csv")

    rows = generate_rows(n)
    write_csv(output_path, rows)

