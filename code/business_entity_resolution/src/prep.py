"""
Stage 1 — preprocessing.

Reads the six raw TSV source files, normalizes business names and addresses,
and writes one Parquet cache per source.  Every later stage reads only the
caches, so normalization runs exactly once.

Run:  python -m src.prep            (from code/business_entity_resolution)
"""
from __future__ import annotations

import csv
import os
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import normalize as N  # noqa: E402

CHUNK = 500_000
COLS = ["entity_id", "business_name", "business_address", "country"]


def _norm_series(series: pd.Series) -> list[str]:
    return [N.norm_text(x) for x in series.fillna("")]


def prep_file(src_tsv: str, out_parquet: str) -> int:
    """Normalize one source TSV into a Parquet file. Returns row count."""
    if os.path.exists(out_parquet):
        n = pq.ParquetFile(out_parquet).metadata.num_rows
        print(f"  [skip] {out_parquet} already exists ({n:,} rows)")
        return n
    t0 = time.time()
    writer = None
    total = 0
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    tmp = out_parquet + ".tmp"
    for chunk in pd.read_csv(
        src_tsv, sep="\t", dtype=str, chunksize=CHUNK,
        quoting=csv.QUOTE_NONE, na_filter=False, encoding="utf-8",
    ):
        chunk = chunk[COLS]
        table = pa.table({
            "entity_id": pa.array(chunk["entity_id"].tolist(), pa.string()),
            "name_norm": pa.array(_norm_series(chunk["business_name"]), pa.string()),
            "addr_norm": pa.array(_norm_series(chunk["business_address"]), pa.string()),
            "country": pa.array(chunk["country"].tolist(), pa.string()),
        })
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table)
        total += len(chunk)
        print(f"    {total:,} rows  ({time.time()-t0:.0f}s)", file=sys.stderr)
    if writer:
        writer.close()
    os.replace(tmp, out_parquet)
    print(f"  wrote {out_parquet} ({total:,} rows, {time.time()-t0:.0f}s)")
    return total


def main(data_root: str = "student_resource/student_resource/dataset",
         cache_dir: str = "work/cache") -> None:
    jobs = []
    for split in ("train", "test"):
        for s in (1, 2, 3):
            jobs.append((f"{data_root}/{split}/{split}_source{s}.tsv",
                         f"{cache_dir}/{split}_source{s}.parquet"))
    for src, dst in jobs:
        print(f"* {src} -> {dst}")
        prep_file(src, dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
