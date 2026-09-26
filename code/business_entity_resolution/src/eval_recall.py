"""
Evaluate candidate-set recall against train ground truth; dump positive pairs.

Reports recall of the candidate set (the recall ceiling for the matcher),
mean candidates per S1, and saves pos_s1.npy / pos_t.npy with the true
(s1_row, target_idx) pairs for training-set construction.

Usage:
  python -m src.eval_recall <cache_dir> <cand_dir> <out_dir> <gt_tsv> <split>
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def target_id_arrays(cache_dir: str, split: str):
    tids2 = []
    for b in pq.ParquetFile(f"{cache_dir}/{split}_source2.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        tids2.extend(b.column("entity_id").to_pylist())
    tids3 = []
    for b in pq.ParquetFile(f"{cache_dir}/{split}_source3.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        tids3.extend(b.column("entity_id").to_pylist())
    return tids2, tids3


def s1_ids_array(cache_dir: str, split: str):
    out = []
    for b in pq.ParquetFile(f"{cache_dir}/{split}_source1.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        out.extend(b.column("entity_id").to_pylist())
    return out


def run(cache_dir: str, cand_dir: str, out_dir: str, gt_tsv: str, split: str) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    flat = np.load(f"{cand_dir}/cand.flat.npy")
    offs = np.load(f"{cand_dir}/cand.offs.npy")
    n_s1 = len(offs) - 1

    tids2, tids3 = target_id_arrays(cache_dir, split)
    n2 = len(tids2)
    t2i = {e: i for i, e in enumerate(tids2)}
    t2i.update({e: i + n2 for i, e in enumerate(tids3)})
    del tids2, tids3

    s1_ids = s1_ids_array(cache_dir, split)
    assert len(s1_ids) == n_s1, (len(s1_ids), n_s1)
    s1_row = {e: i for i, e in enumerate(s1_ids)}

    gt = {}
    with open(gt_tsv, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for s1, m in r:
            ids = [x for x in m.split(",") if x]
            if ids:
                gt[s1] = ids
    print(f"GT: {len(gt):,} S1 with >=1 match ({time.time()-t0:.0f}s)", file=sys.stderr)

    # mark candidate hits
    is_hit = np.zeros(n_s1 + 1, dtype=np.int64)   # prefix over pairs
    hit_cnt = np.zeros(n_s1, dtype=np.int32)
    pos_s1_parts, pos_t_parts = [], []
    s1_of_pair = np.repeat(np.arange(n_s1, dtype=np.int64), np.diff(offs))
    tid_of_pair = np.empty(len(flat), dtype=np.int64)
    # vectorized target-id -> local index needs a lookup over 10M strings:
    # do it in chunks via dict (fast enough in C-level loop over python list)
    flat_list = flat.tolist()
    # build row->tid array lazily: we need entity id strings per target row;
    # instead compare by row index: build gt as row indices
    gt_rows = {}
    for s1, ids in gt.items():
        i1 = s1_row.get(s1)
        if i1 is None:
            continue
        rows = []
        for e in ids:
            i = t2i.get(e)
            if i is not None:
                rows.append(i)
        if rows:
            gt_rows[i1] = rows
    print(f"GT rows mapped: {len(gt_rows):,} ({time.time()-t0:.0f}s)", file=sys.stderr)

    # is target row a true match of its s1? vectorized via sorted (s1,t) keys
    pos_s1, pos_t = [], []
    hits = 0
    total_gt = sum(len(rows) for rows in gt_rows.values())
    gt_keys = []
    for i1, rows in gt_rows.items():
        for r in rows:
            gt_keys.append((np.int64(i1) << np.int64(32)) | np.int64(r))
    gt_keys = np.asarray(gt_keys, dtype=np.int64)
    gt_keys.sort()
    s1_of_pair = np.repeat(np.arange(n_s1, dtype=np.int64), np.diff(offs))
    pair_keys = (np.asarray(flat, dtype=np.int64) << np.int64(32)) | s1_of_pair
    del s1_of_pair
    is_pos = np.isin(pair_keys, gt_keys, assume_unique=False)
    del pair_keys, gt_keys
    hits = int(is_pos.sum())
    pos_s1 = s1_of_pair_orig = None  # keep names quiet
    pos_idx = np.flatnonzero(is_pos)
    del is_pos
    pos_s1_arr = np.repeat(np.arange(n_s1, dtype=np.int64), np.diff(offs))[pos_idx]
    pos_t_arr = np.asarray(flat)[pos_idx]
    recall = hits / max(total_gt, 1)
    print(f"candidate recall = {recall:.4f}  ({hits:,}/{total_gt:,} true pairs in candidates)",
          file=sys.stderr)
    mean_cand = len(flat) / n_s1
    print(f"mean candidates/S1 = {mean_cand:.1f}", file=sys.stderr)

    np.save(f"{out_dir}/pos_s1.npy", pos_s1_arr)
    np.save(f"{out_dir}/pos_t.npy", pos_t_arr)
    with open(f"{out_dir}/recall.json", "w", encoding="utf-8") as f:
        import json
        json.dump({"recall": recall, "n_true_pairs": int(total_gt),
                   "n_pos_in_cand": int(hits), "mean_cand": mean_cand}, f, indent=2)
    print(f"* wrote positives ({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2], a[3], a[4], a[5])
