import os
import csv
import time
import random as rand
import multiprocessing as mp
from typing import Any, List, Tuple, Union
import numpy as np
from fractions import Fraction
from collections import Counter

HEADER = [
    "set", "add_ds_card", "diff_ds_card", "mult_ds_card",
    "set_cardinality", "diameter", "density", "dc",
    "is_ap", "is_gp", "add_energy", "mult_energy"
]

MIN_HEADER = [
    "set", "add_ds_card", "mult_ds_card"
]


def _mask_to_subset(mask: int, n: int) -> np.ndarray:
    # return a numpy array of Python `int` objects (dtype=object)
    vals = [i + 1 for i in range(n) if (mask >> i) & 1]
    return np.array(vals, dtype=object)


def _compute_row(subset: np.ndarray, mask: int) -> List[Any]:
    A = list(subset)

    # pairwise sums, diffs, products
    sums = {a + b for a in A for b in A}
    prods = {a * b for a in A for b in A}

    add_ds_card = len(sums)
    mult_ds_card = len(prods)

    return [mask, add_ds_card, mult_ds_card]

def _worker(args) -> str:
    chunk_id, n, k, flush_every, out_dir = args
    total = 1 << n
    file_id = chunk_id + 1
    path = os.path.join(out_dir, f"set_info_{n}_{file_id:04d}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)

        buf: list[list] = []
        for mask in range(chunk_id, total, k):
            if mask == 0:
                continue
            subset = _mask_to_subset(mask, n)
            buf.append(_compute_row(subset, mask))

            if len(buf) >= flush_every:
                w.writerows(buf)
                buf.clear()

        if buf:
            w.writerows(buf)
            buf.clear()

    return path


def compute_powerset(n: int, out_dir: str, jobs: int, k: int, flush_every: int, mp_context: str = "fork") -> None:
    if n < 1:
        raise ValueError("n must be >= 1")
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if flush_every < 1:
        raise ValueError("flush_every must be >= 1")

    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    try:
        ctx = mp.get_context(mp_context)
    except ValueError:
        ctx = mp.get_context()

    total_tasks = k * jobs
    tasks = [(i, n, total_tasks, flush_every, out_dir) for i in range(total_tasks)]

    with ctx.Pool(processes=jobs) as pool:
        done = 0
        for path in pool.imap_unordered(_worker, tasks, chunksize=1):
            done += 1
            print(f"{(100*done)//(total_tasks)}% done, wrote {path}, {time.time()-t0:.1f}s since start")
