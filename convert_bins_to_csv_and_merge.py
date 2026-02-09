#!/usr/bin/env python3
"""Batch-convert SPP1 .bin files to CSV and create a merged CSV.

The .bin layout matches convert_data.py:
  Header: <8sBBHI  (magic[8], version(u8), n(u8), max_sum(u16), record_cnt(u32))
  Record: <BBHQ    (k(u8), add(u8), mult(u16), count(u64))

This script never modifies input files; it only reads .bin files and writes CSV outputs.
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, TextIO

HDR = struct.Struct("<8sBBHI")  # little-endian
REC = struct.Struct("<BBHQ")

MAGIC = b"SPP1BIN\x00"
VERSION = 1


def iter_bin_files(in_dir: Path, pattern: str) -> list[Path]:
    if not in_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {in_dir}")
    files = sorted(p for p in in_dir.glob(pattern) if p.is_file())
    if not files:
        raise ValueError(f"No files matched {pattern!r} in {in_dir}")
    return files


def process_bin_file(
    path: Path,
    merged_counts: DefaultDict[tuple[int, int, int], int],
    per_file_csv: TextIO | None = None,
) -> tuple[int, int, int]:
    """Read one .bin file, update merged_counts, and optionally write per-file CSV.

    Returns (n, max_sum, rows_read).
    """
    with path.open("rb") as f:
        hdr = f.read(HDR.size)
        if len(hdr) != HDR.size:
            raise ValueError(f"File too small: {path}")

        magic, ver, n, max_sum, rcnt = HDR.unpack(hdr)
        if magic != MAGIC:
            raise ValueError(f"Bad magic in {path}: {magic!r}")
        if ver != VERSION:
            raise ValueError(f"Unsupported version in {path}: {ver}")

        rows_read = 0
        if per_file_csv is not None:
            per_file_csv.write("set_cardinality,add_ds_card,mult_ds_card,count\n")

        for _ in range(int(rcnt)):
            buf = f.read(REC.size)
            if len(buf) != REC.size:
                raise ValueError(f"Truncated file: {path}")
            k, add, mult, cnt = REC.unpack(buf)
            key = (int(k), int(add), int(mult))
            merged_counts[key] += int(cnt)
            rows_read += 1
            if per_file_csv is not None:
                per_file_csv.write(f"{key[0]},{key[1]},{key[2]},{int(cnt)}\n")

        return int(n), int(max_sum), rows_read


def write_csv(out_path: Path, rows: Iterable[tuple[int, int, int, int]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("set_cardinality,add_ds_card,mult_ds_card,count\n")
        for k, add, mult, cnt in rows:
            f.write(f"{k},{add},{mult},{cnt}\n")


def merged_rows_from_counts(counts: dict[tuple[int, int, int], int]) -> list[tuple[int, int, int, int]]:
    return [(k, add, mult, counts[(k, add, mult)]) for (k, add, mult) in sorted(counts.keys())]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Convert all SPP1 .bin files in a directory to CSV and produce a merged CSV.",
    )
    ap.add_argument("in_dir", type=Path, help="Directory containing .bin files")
    ap.add_argument(
        "out_merged_csv",
        type=Path,
        help="Path to write merged CSV (will be created)",
    )
    ap.add_argument(
        "--pattern",
        default="*.bin",
        help="Glob pattern to match input files (default: *.bin)",
    )
    ap.add_argument(
        "--per-file-dir",
        type=Path,
        default=None,
        help="If set, also write one CSV per input into this directory",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSVs",
    )

    args = ap.parse_args(argv)

    in_files = iter_bin_files(args.in_dir, args.pattern)

    if args.out_merged_csv.exists() and not args.overwrite:
        print(f"Refusing to overwrite existing: {args.out_merged_csv} (use --overwrite)", file=sys.stderr)
        return 2

    if args.per_file_dir is not None and args.per_file_dir.exists() and not args.per_file_dir.is_dir():
        print(f"per-file-dir exists but is not a directory: {args.per_file_dir}", file=sys.stderr)
        return 2

    expected_n: int | None = None
    expected_max_sum: int | None = None

    merged_counts: DefaultDict[tuple[int, int, int], int] = defaultdict(int)
    total_rows = 0

    for idx, path in enumerate(in_files, start=1):
        per_file_fh: TextIO | None = None
        out_csv: Path | None = None
        if args.per_file_dir is not None:
            out_csv = args.per_file_dir / (path.stem + ".csv")
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            if out_csv.exists() and not args.overwrite:
                print(f"Skipping existing per-file CSV: {out_csv}", file=sys.stderr)
            else:
                per_file_fh = out_csv.open("w", encoding="utf-8")

        try:
            n, max_sum, rows_read = process_bin_file(path, merged_counts, per_file_fh)
        finally:
            if per_file_fh is not None:
                per_file_fh.close()

        if expected_n is None:
            expected_n = n
            expected_max_sum = max_sum
        else:
            if n != expected_n or max_sum != expected_max_sum:
                raise ValueError(
                    f"Inconsistent header in {path}: n={n}, max_sum={max_sum} (expected n={expected_n}, max_sum={expected_max_sum})"
                )

        total_rows += rows_read

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(in_files)} files...", file=sys.stderr)

    merged_rows = merged_rows_from_counts(dict(merged_counts))
    write_csv(args.out_merged_csv, merged_rows)

    print(f"Read {len(in_files)} files; merged {total_rows} rows into {len(merged_rows)} unique keys -> {args.out_merged_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
