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

    # map GT ids -> row indices (s1_row for S1; n2-offset for S2/S3 concat)
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

    # is target row a true match of its s1? vectorized via sorted (s1,t) keys.
    # pair key = (s1_row << 32) | target_idx; compare against sorted gt_keys
    # in chunks with searchsorted -> O(1) temp memory, no giant isin.
    gt_keys = []
    for i1, rows in gt_rows.items():
        for r in rows:
            gt_keys.append((np.int64(i1) << np.int64(32)) | np.int64(r))
    gt_keys = np.unique(np.asarray(gt_keys, dtype=np.int64))
    total_gt = len(gt_keys)

    pos_s1_parts, pos_t_parts = [], []
    hits = 0
    for start in range(0, n_s1, 20_000):
        end = min(start + 20_000, n_s1)
        seg = flat[offs[start]:offs[end]]
        s1_of_seg = np.repeat(
            np.arange(start, end, dtype=np.int64), np.diff(offs[start:end + 1]))
        keys = (s1_of_seg << np.int64(32)) | np.asarray(seg, dtype=np.int64)
        j = np.searchsorted(gt_keys, keys)
        j_c = np.minimum(j, len(gt_keys) - 1)
        is_pos = gt_keys[j_c] == keys
        hits += int(is_pos.sum())
        if is_pos.any():
            idx = np.flatnonzero(is_pos)
            pos_s1_parts.append(s1_of_seg[idx])
            pos_t_parts.append(seg[idx])
    pos_s1_arr = (np.concatenate(pos_s1_parts) if pos_s1_parts
                  else np.zeros(0, dtype=np.int64))
    pos_t_arr = (np.concatenate(pos_t_parts) if pos_t_parts
                 else np.zeros(0, dtype=np.int32))
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
