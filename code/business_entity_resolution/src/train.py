"""
Stage 4 driver - train the matcher and calibrate the F0.5 threshold.

Pipeline:
 1. split train S1 rows into train/val halves (deterministic hash split)
 2. labels: pair is positive iff (s1_row, target) is a ground-truth pair
 3. LightGBM binary classifier on pair features
 4. threshold sweep on validation pairs -> macro-F0.5
 5. persist booster + threshold + metrics

Usage:
  python -m src.train <feature_dir> <cand_dir> <gt_tsv> <cache_dir> <out_dir>
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metric import evaluate_groups  # noqa: E402


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


def main(feature_dir: str, cand_dir: str, gt_tsv: str, cache_dir: str,
         out_dir: str) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    import lightgbm as lgb

    X = np.load(f"{feature_dir}/X.npy", mmap_mode="r")
    pair_s1 = np.load(f"{feature_dir}/pair_s1.npy")
    pair_t = np.load(f"{feature_dir}/pair_t.npy")
    n_pairs = X.shape[0]
    print(f"* {n_pairs:,} pairs, {X.shape[1]} features", file=sys.stderr)

    truth, n2 = build_truth_rows(cache_dir, gt_tsv)

    # deterministic 50/50 S1 split
    rng = np.random.default_rng(42)
    all_s1 = np.unique(pair_s1)
    is_val_s1 = rng.random(len(all_s1)) < 0.5
    val_s1 = set(all_s1[is_val_s1].tolist())
    in_val = np.isin(pair_s1, np.fromiter(val_s1, dtype=np.int64))
    y = np.zeros(n_pairs, dtype=np.int8)
    pos_mask = np.zeros(n_pairs, dtype=bool)
    # mark positives
    ts = pair_t
    ss = pair_s1
    # vectorized: build set of (s1,t) via truth dict
    # (python loop; 1-2 min for 500M pairs is too slow -> use dict of arrays)
    truth_first = {}
    for i1, rows in truth.items():
        truth_first[i1] = np.fromiter(rows, dtype=np.int64, count=len(rows))
    # pairs where target in truth[s1]: use per-pair lookup via searchsorted on
    # concatenated (s1<<32 | t) keys
    keys_truth = []
    for i1, arr in truth_first.items():
        keys_truth.append((arr << np.int64(32)) | np.int64(i1))
    keys_truth = np.concatenate(keys_truth) if keys_truth else np.empty(0, np.int64)
    keys_truth.sort()
    keys_pairs = (ts.astype(np.int64) << np.int64(32)) | ss.astype(np.int64)
    pos_mask = np.isin(keys_pairs, keys_truth, assume_unique=False)
    y = pos_mask.astype(np.int8)
    print(f"  positives: {y.sum():,} ({y.mean()*100:.2f}%) ({time.time()-t0:.0f}s)",
          file=sys.stderr)

    tr_mask = ~in_val
    Xtr, ytr = X[tr_mask], y[tr_mask]
    Xva, yva = X[in_val], y[in_val]
    s1_va = pair_s1[in_val]
    t_va = pair_t[in_val]
    print(f"  train pairs={len(ytr):,} val pairs={len(yva):,} "
          f"({time.time()-t0:.0f}s)", file=sys.stderr)

    pos_w = max((ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0)
    params = {
        "objective": "binary",
        "learning_rate": 0.06,
        "num_leaves": 96,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "scale_pos_weight": min(pos_w, 50.0),
        "verbose": -1,
        "seed": 7,
    }
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=600,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    booster.save_model(f"{out_dir}/model.txt")

    pred_va = booster.predict(Xva, num_iteration=booster.best_iteration)
    truth_val = {s: truth[s] for s in val_s1 if s in truth}
    n_s1_all = int(pair_s1.max()) + 1
    best = (-1.0, None, None)
    for thr in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        f, st = evaluate_groups(s1_va, t_va, pred_va, truth_val, n_s1_all, thr)
        print(f"  thr={thr:.2f}  F0.5={f:.4f}  {st}", file=sys.stderr)
        if f > best[0]:
            best = (f, thr, st)
    # refine
    lo, hi = max(best[1] - 0.1, 0.01), best[1] + 0.1
    for thr in np.arange(lo, hi, 0.02):
        f, st = evaluate_groups(s1_va, t_va, pred_va, truth_val, n_s1_all, float(thr))
        if f > best[0]:
            best = (f, float(thr), st)
    print(f"  BEST thr={best[1]:.3f} F0.5={best[0]:.4f}", file=sys.stderr)
    with open(f"{out_dir}/threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best[1], "val_f05": best[0], "stats": best[2],
                   "best_iteration": int(booster.best_iteration or 0),
                   "n_train_pairs": int(len(ytr)), "n_val_pairs": int(len(yva))},
                  f, indent=2)
    print(f"* done ({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], a[4], a[5])
