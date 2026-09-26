"""
Compact per-source text blobs for fast feature computation.

For each source we persist one UTF-8 .bin file per text field (name, addr)
plus int64 offsets, so workers can memory-map gigabytes of text and decode
only the slices they touch.  Also stores:

* country.npy   int8 country code (hash of the country string, open-set)
* digits.npy    digit-signature blob + offsets ("|".join(sorted digit runs))
* postal.npy    int64 hash of the 5/6-digit postal code (0 if none)
* nkeys.npy     int32 number of blocking keys per record
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import normalize as N  # noqa: E402
from src import keys as K  # noqa: E402


def _write_blob(values: list[str], out_bin: str, out_off: str) -> None:
    buf = bytearray()
    offs = np.zeros(len(values) + 1, dtype=np.int64)
    for i, s in enumerate(values):
        b = s.encode("utf-8")
        buf += b
        offs[i + 1] = len(buf)
    with open(out_bin, "wb") as f:
        f.write(bytes(buf))
    np.save(out_off, offs)


def build_source(cache_parquet: str, keys_dir: str, out_dir: str,
                 split: str, s: int) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    pf = pq.ParquetFile(cache_parquet)
    names, addrs, countries = [], [], []
    for b in pf.iter_batches(batch_size=250_000,
                             columns=["name_norm", "addr_norm", "country"]):
        names.extend(b.column("name_norm").to_pylist())
        addrs.extend(b.column("addr_norm").to_pylist())
        countries.extend(b.column("country").to_pylist())
    n = len(names)
    _write_blob(names, f"{out_dir}/{split}_s{s}.name.bin",
                f"{out_dir}/{split}_s{s}.name.off.npy")
    _write_blob(addrs, f"{out_dir}/{split}_s{s}.addr.bin",
                f"{out_dir}/{split}_s{s}.addr.off.npy")

    # country codes: open-set hash into int8 buckets by first-seen order
    cc = {}
    ctry = np.empty(n, dtype=np.int8)
    for i, c in enumerate(countries):
        j = cc.get(c)
        if j is None:
            j = len(cc)
            cc[c] = j
        ctry[i] = j
    np.save(f"{out_dir}/{split}_s{s}.country.npy", ctry)
    with open(f"{out_dir}/{split}_s{s}.country.json", "w", encoding="utf-8") as f:
        json.dump(cc, f)

    # digit signatures + postal
    dig = ["|".join(sorted(set(N.numeric_tokens(a)))) for a in addrs]
    _write_blob(dig, f"{out_dir}/{split}_s{s}.dig.bin",
                f"{out_dir}/{split}_s{s}.dig.off.npy")
    postal = np.zeros(n, dtype=np.int64)
    for i, d in enumerate(dig):
        for run in d.split("|"):
            if len(run) in (5, 6):
                postal[i] = hash(("po", run)) & 0x7FFFFFFFFFFFFFFF
                break
    np.save(f"{out_dir}/{split}_s{s}.postal.npy", postal)

    # blocking-key counts (from keys dir)
    _, offs = K.load_keys(keys_dir, split, s)
    np.save(f"{out_dir}/{split}_s{s}.nkeys.npy", np.diff(offs).astype(np.int32))
    print(f"  {split}_s{s}: {n:,} records ({time.time()-t0:.0f}s)", file=sys.stderr)


def main(cache_dir: str, keys_dir: str, out_dir: str,
         splits=("train", "test")) -> None:
    for split in splits:
        for s in (1, 2, 3):
            marker = f"{out_dir}/{split}_s{s}.nkeys.npy"
            if os.path.exists(marker):
                print(f"  [skip] {split}_s{s}")
                continue
            print(f"* building blobs for {split}_source{s}")
            build_source(f"{cache_dir}/{split}_source{s}.parquet", keys_dir,
                         out_dir, split, s)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], tuple(a[4].split(",")) if len(a) > 4 else ("train", "test"))
