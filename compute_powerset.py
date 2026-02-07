import os
import csv
import time
import ctypes
import multiprocessing as mp
from collections import Counter
from typing import Dict, Tuple


# ----------------------------
# C library binding
# ----------------------------

def load_sumprod_lib(so_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(so_path)

    # int sp_compute_from_mask(uint64_t mask, int n, int *out_add_card, int *out_mult_card)
    lib.sp_compute_from_mask.argtypes = [
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.sp_compute_from_mask.restype = ctypes.c_int
    return lib


# ----------------------------
# Worker
# ----------------------------

def _worker(args) -> str:
    """
    Each worker:
      - iterates masks in a strided partition
      - calls the C function to get (|A+A|, |A*A|)
      - aggregates into Counter[(add,mult)] += 1
      - writes a small CSV partial: add,mult,count
    """
    chunk_id, n, total_tasks, so_path, out_dir = args

    lib = load_sumprod_lib(so_path)

    total = 1 << n
    out_path = os.path.join(out_dir, f"pairs_{n}_{chunk_id+1:04d}.csv")

    counts: Counter[Tuple[int, int]] = Counter()

    add_out = ctypes.c_int()
    mul_out = ctypes.c_int()

    for mask in range(chunk_id, total, total_tasks):
        if mask == 0:
            continue

        rc = lib.sp_compute_from_mask(
            ctypes.c_uint64(mask),
            ctypes.c_int(n),
            ctypes.byref(add_out),
            ctypes.byref(mul_out),
        )
        if rc != 0:
            raise RuntimeError(f"C sp_compute_from_mask failed: rc={rc}, n={n}, mask={mask}")

        counts[(int(add_out.value), int(mul_out.value))] += 1

    # write partial
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["add_ds_card", "mult_ds_card", "count"])
        for (a, m), c in counts.items():
            w.writerow([a, m, c])

    return out_path


# ----------------------------
# Merge partials into final dict
# ----------------------------

def merge_partials(partial_paths) -> Dict[Tuple[int, int], int]:
    merged: Dict[Tuple[int, int], int] = {}
    for path in partial_paths:
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header is None:
                continue
            for row in r:
                a = int(row[0]); m = int(row[1]); c = int(row[2])
                merged[(a, m)] = merged.get((a, m), 0) + c
    return merged


# ----------------------------
# Main entry point
# ----------------------------

def compute_powerset_sumprod_distribution(
    n: int,
    out_dir: str,
    jobs: int,
    k: int,
    so_path: str,
    mp_context: str = "fork",
    write_merged_csv: bool = True,
) -> Dict[Tuple[int, int], int]:
    """
    Computes the distribution:
        D[(|A+A|, |A*A|)] = #{ A ⊆ [n], A != ∅ : (|A+A|,|A*A|) equals that pair }

    Parallelization matches your original structure:
      total_tasks = k * jobs
      each task is a strided mask-walk
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n > 63:
        raise ValueError("n must be <= 63 for uint64 mask encoding")
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    if not os.path.exists(so_path):
        raise FileNotFoundError(f"Shared library not found: {so_path}")

    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    try:
        ctx = mp.get_context(mp_context)
    except ValueError:
        ctx = mp.get_context()

    total_tasks = k * jobs
    tasks = [(i, n, total_tasks, so_path, out_dir) for i in range(total_tasks)]

    partial_paths = []
    with ctx.Pool(processes=jobs) as pool:
        done = 0
        for path in pool.imap_unordered(_worker, tasks, chunksize=1):
            partial_paths.append(path)
            done += 1
            print(f"{(100*done)//total_tasks}% done, wrote {path}, {time.time()-t0:.1f}s since start")

    dist = merge_partials(partial_paths)

    if write_merged_csv:
        merged_path = os.path.join(out_dir, f"pairs_merged_{n}.csv")
        with open(merged_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["add_ds_card", "mult_ds_card", "count"])
            for (a, m), c in sorted(dist.items()):
                w.writerow([a, m, c])
        print(f"Wrote merged distribution: {merged_path}")

    return dist


# Example usage:
# dist = compute_powerset_sumprod_distribution(
#     n=20,
#     out_dir="./out_pairs_n20",
#     jobs=16,
#     k=2,
#     so_path="./sumprod_mask.so",
# )
# print(len(dist), "distinct (|A+A|,|A*A|) pairs")
