#!/usr/bin/env python3
import struct
import sys
from pathlib import Path

HDR = struct.Struct("<8sBBHI")    # magic[8], version, n, max_sum, record_cnt  (little-endian)
REC = struct.Struct("<BBHQ")      # k, add, mult(u16), count(u64)

def convert(in_path: Path, out_path: Path) -> None:
    data = in_path.read_bytes()
    if len(data) < HDR.size:
        raise ValueError("File too small")

    magic, ver, n, max_sum, rcnt = HDR.unpack_from(data, 0)
    if magic != b"SPP1BIN\x00":
        raise ValueError(f"Bad magic: {magic!r}")
    if ver != 1:
        raise ValueError(f"Unsupported version: {ver}")

    off = HDR.size
    need = off + rcnt * REC.size
    if len(data) < need:
        raise ValueError("Truncated file")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("set_cardinality,add_ds_card,mult_ds_card,count\n")
        for _ in range(rcnt):
            k, add, mult, cnt = REC.unpack_from(data, off)
            off += REC.size
            f.write(f"{k},{add},{mult},{cnt}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} IN.bin OUT.csv", file=sys.stderr)
        sys.exit(2)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))

