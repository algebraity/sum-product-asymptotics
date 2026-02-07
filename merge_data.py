import os, csv, sys
from collections import Counter

d = sys.argv[1] if len(sys.argv) > 1 else "."
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "pairs_merged.csv")

cnt = Counter()
for fn in os.listdir(d):
    if fn.startswith("pairs_") and fn.endswith(".csv") and "merged" not in fn:
        with open(os.path.join(d, fn), newline="", encoding="utf-8") as f:
            r = csv.reader(f); next(r, None)
            for a, m, c in r:
                cnt[(int(a), int(m))] += int(c)

with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["add_ds_card", "mult_ds_card", "count"])
    for (a, m), c in sorted(cnt.items()):
        w.writerow([a, m, c])

