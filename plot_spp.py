#!/usr/bin/env python3
from __future__ import annotations

"""plot_nspp.py

Plot NSPP-style scatter plots from aggregated counts keyed by:
    set_cardinality, add_ds_card, mult_ds_card, count

Two modes:
    --normalized      (default) matches O'Bryant et al.:
            x = K_k(|A*A|), y = K_k(|A+A|), both mapped into [1,2].
    --non-normalized  plots raw values:
            x = |A*A| (mult_ds_card), y = |A+A| (add_ds_card).

Usage:
    python3 plot_nspp.py --input data.csv --kmin 3 --kmax 32 --output nspp_3_32.png

Notes:
- Expects CSV header: set_cardinality,add_ds_card,mult_ds_card,count
- Designed for aggregated output; it does NOT expand points by count.
"""

import argparse
import math
from pathlib import Path

# Heavy dependencies (numpy/pandas/matplotlib) are imported inside main()
# so `--help` works even if they are not installed.


def K_params(k: int) -> tuple[float, float]:
    """
    K_k(x) = log_k(x) + m_k * x + b_k, with
      K_k(2k-1) = 1  and  K_k(k(k+1)/2) = 2.
    Defined for k >= 3.
    """
    if k < 3:
        raise ValueError("K_k is defined for k >= 3 in this normalization.")
    x1 = 2 * k - 1
    x2 = k * (k + 1) / 2.0
    ln_k = math.log(k)
    L1 = math.log(x1) / ln_k
    L2 = math.log(x2) / ln_k
    m = (1.0 - L2 + L1) / (x2 - x1)
    b = 1.0 - L1 - m * x1
    return m, b


def K_of(k: int, x: np.ndarray, m: float, b: float) -> np.ndarray:
    # x is positive here (|A+A|>=1, |A*A|>=1)
    return np.log(x) / math.log(k) + m * x + b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV (merged counts).")
    ap.add_argument("--output", required=True, help="Output image file (png/pdf/svg).")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=32)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--normalized",
        dest="normalized",
        action="store_true",
        help="Normalize axes to [1,2] using K_k (default; matches current behavior).",
    )
    mode.add_argument(
        "--non-normalized",
        dest="normalized",
        action="store_false",
        help="Plot raw |A*A| vs |A+A| without K_k normalization.",
    )
    ap.set_defaults(normalized=True)
    ap.add_argument("--chunksize", type=int, default=2_000_000,
                    help="Read CSV in chunks to reduce peak memory.")
    ap.add_argument("--no_size_weight", action="store_true",
                    help="If set, do not scale marker size by log(count).")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing plotting dependencies. Install with: python3 -m pip install numpy pandas matplotlib\n"
            f"Details: {e}"
        )

    globals()["np"] = np
    globals()["pd"] = pd
    globals()["plt"] = plt

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    kmin, kmax = args.kmin, args.kmax
    if kmin < 1 or kmax < kmin:
        raise ValueError("Bad k-range.")

    # Read only the needed rows/types.
    dtypes = {
        "set_cardinality": "uint8",
        "add_ds_card": "uint16",
        "mult_ds_card": "uint16",
        "count": "uint64",
    }

    kept = []
    for chunk in pd.read_csv(in_path, dtype=dtypes, chunksize=args.chunksize):
        chunk = chunk[(chunk["set_cardinality"] >= kmin) & (chunk["set_cardinality"] <= kmax)]
        if not chunk.empty:
            kept.append(chunk)

    if not kept:
        raise RuntimeError(f"No rows in k-range [{kmin},{kmax}]")

    df = pd.concat(kept, ignore_index=True)

    ks = np.sort(df["set_cardinality"].unique())

    fig, ax = plt.subplots(figsize=(8, 8), dpi=args.dpi)

    if args.normalized:
        # Precompute K_k parameters for each k used.
        params = {int(k): K_params(int(k)) for k in ks}

        # Prepare figure (square, [1,2]^2).
        ax.set_xlim(1.0, 2.0)
        ax.set_ylim(1.0, 2.0)
        ax.set_aspect("equal", adjustable="box")

        ticks = np.linspace(1.0, 2.0, 11)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("|A+A|", fontsize=14)
        ax.set_ylabel("|A*A|", fontsize=14)

    # Match the paper’s “smaller k on top” feel by drawing large k first.
    # Use a rainbow-like map for k.
    cmap = plt.get_cmap("rainbow")

    # Plot per-k to keep a single PathCollection per k (much faster than per-row plotting).
    # Also lets us set colors cleanly and control layering.
    for k in sorted((int(x) for x in ks), reverse=True):
        sub = df[df["set_cardinality"] == k]
        if sub.empty:
            continue
        add = sub["add_ds_card"].to_numpy(dtype=np.float64)
        mult = sub["mult_ds_card"].to_numpy(dtype=np.float64)
        cnt = sub["count"].to_numpy(dtype=np.uint64)

        if args.normalized:
            m, b = params[k]
            x = K_of(k, add, m, b)   # x-axis: sums (AP)
            y = K_of(k, mult,  m, b)   # y-axis: products (GP)
        else:
            x = add
            y = mult

        # Color by k (constant within this batch)
        cval = np.full(x.shape, k, dtype=np.float32)

        ax.scatter(
            x, y,
            c=cval,
            cmap=cmap,
            vmin=kmin,
            vmax=kmax,
            s=100,
            marker=".",
            linewidths=0,
            alpha=0.6,
            rasterized=True, 
        )

    # Colorbar labeled by |A|.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=kmin, vmax=kmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|A|", fontsize=12)

    if args.normalized:
        ax.set_title(f"NSPP[{kmin},{kmax}] from all subsets of [45] (aggregated)", fontsize=14)
    else:
        # Set limits from the raw data for a sensible view.
        y_min = float(df["mult_ds_card"].min())
        y_max = float(df["mult_ds_card"].max())
        x_min = float(df["add_ds_card"].min())
        x_max = float(df["add_ds_card"].max())
        # Small padding to avoid points on the border.
        y_pad = max(1.0, 0.02 * (y_max - y_min))
        x_pad = max(1.0, 0.02 * (x_max - x_min))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Raw NSPP[{kmin},{kmax}] from all subsets of [45] (aggregated)", fontsize=14)

    out_path = Path(args.output)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()

