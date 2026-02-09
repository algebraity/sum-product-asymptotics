import os, csv, sys
from collections import Counter

n = input("Enter n: ")

d = sys.argv[1] if len(sys.argv) > 1 else "."

total = 0
for fn in os.listdir(d):
    if fn.endswith(".csv"):
        with open(os.path.join(d, fn), newline="", encoding="utf-8") as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if not row[3] == "count":
                    total += int(row[3])

print(total == (2**int(n) - 1))

