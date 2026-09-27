"""
Build truth dictionary from ground truth TSV file.
"""
from __future__ import annotations

import csv
import os

import numpy as np


def build_truth_rows(cache_dir: str, gt_tsv: str) -> tuple[dict, int]:
    """GT as {s1_row: set(target_row)} using source1 row order + S2/S3 concat."""
    import pyarrow.parquet as pq
    tids2 = []
    for b in pq.ParquetFile(f"{cache_dir}/train_source2.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        tids2.extend(b.column("entity_id").to_pylist())
    n2 = len(tids2)
    t2i = {e: i for i, e in enumerate(tids2)}
    del tids2
    for b in pq.ParquetFile(f"{cache_dir}/train_source3.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        for e in b.column("entity_id").to_pylist():
            t2i[e] = len(t2i) + n2 if False else 0  # placeholder, fill below
    # rebuild properly
    tids3 = []
    for b in pq.ParquetFile(f"{cache_dir}/train_source3.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        tids3.extend(b.column("entity_id").to_pylist())
    for i, e in enumerate(tids3):
        t2i[e] = n2 + i
    s1_ids = []
    for b in pq.ParquetFile(f"{cache_dir}/train_source1.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        s1_ids.extend(b.column("entity_id").to_pylist())
    s1_row = {e: i for i, e in enumerate(s1_ids)}

    truth = {}
    with open(gt_tsv, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for s1, m in r:
            i1 = s1_row.get(s1)
            if i1 is None:
                continue
            rows = set()
            for e in m.split(","):
                i = t2i.get(e)
                if i is not None:
                    rows.add(i)
            if rows:
                truth[i1] = rows
    return truth, n2