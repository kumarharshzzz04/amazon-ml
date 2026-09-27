"""
Stage 4 driver - train the LightGBM pair classifier.

Inputs (feature_dir, from src.build_features_sub): X.npy (memmap), pair_s1.npy,
pair_t.npy, meta.json.  Labels come straight from the ground-truth positives
dumped by src.eval_recall (pos_s1.npy / pos_t.npy in cand_dir).

A deterministic ~50/50 S1-row split gives train/val; a small subsample of the
val pairs (all val positives + capped negatives) drives early stopping.  Final
F0.5 threshold calibration happens separately on full candidate pools
(src.calibrate) so the score distribution matches inference.

Usage:
  python -m src.train <feature_dir> <cand_dir> <out_dir>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(feature_dir: str, cand_dir: str, out_dir: str) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    import lightgbm as lgb

    X = np.load(f"{feature_dir}/X.npy", mmap_mode="r")
    pair_s1 = np.load(f"{feature_dir}/pair_s1.npy")
    pair_t = np.load(f"{feature_dir}/pair_t.npy")
    n_pairs, nf = X.shape
    n_s1 = int(pair_s1.max()) + 1
    print(f"* {n_pairs:,} pairs, {nf} features ({time.time()-t0:.0f}s)",
          file=sys.stderr)

    # labels: (s1, t) key match against gt positives
    pos_s1 = np.load(f"{cand_dir}/pos_s1.npy")
    pos_t = np.load(f"{cand_dir}/pos_t.npy").astype(np.int64)
    pos_keys = np.unique((pos_s1 << np.int64(32)) | pos_t)
    del pos_s1, pos_t
    keys = (pair_s1 << np.int64(32)) | pair_t
    j = np.minimum(np.searchsorted(pos_keys, keys), len(pos_keys) - 1)
    y = (pos_keys[j] == keys).astype(np.int8)
    del keys, j, pos_keys
    print(f"  positives: {y.sum():,} ({y.mean()*100:.2f}%) "
          f"({time.time()-t0:.0f}s)", file=sys.stderr)

    # deterministic 50/50 S1-row split
    rng = np.random.default_rng(42)
    val_mask = np.zeros(n_s1, dtype=bool)
    all_s1 = np.unique(pair_s1)
    val_mask[all_s1[rng.random(len(all_s1)) < 0.5]] = True
    in_val = val_mask[pair_s1]
    tr_idx = np.flatnonzero(~in_val)
    va_idx = np.flatnonzero(in_val)
    del in_val

    # shrink val to <=1.5M pairs (keep every val positive)
    va_pos = va_idx[y[va_idx] == 1]
    va_neg = va_idx[y[va_idx] == 0]
    if len(va_neg) > 1_500_000 - len(va_pos):
        keep_n = max(1_500_000 - len(va_pos), 100_000)
        va_neg = va_neg[rng.choice(len(va_neg), size=keep_n, replace=False)]
    va_idx = np.sort(np.concatenate([va_pos, va_neg]))
    print(f"  train={len(tr_idx):,}  val={len(va_idx):,} "
          f"(pos {len(va_pos):,}) ({time.time()-t0:.0f}s)", file=sys.stderr)

    Xtr = np.ascontiguousarray(X[tr_idx])
    ytr = y[tr_idx]
    Xva = np.ascontiguousarray(X[va_idx])
    yva = y[va_idx]

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
        "num_threads": 10,
    }
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=600,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50),
                                   lgb.log_evaluation(100)])
    booster.save_model(f"{out_dir}/model.txt")

    with open(f"{out_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump({"best_iteration": int(booster.best_iteration or 0),
                   "n_features": int(nf),
                   "n_train_pairs": int(len(ytr)),
                   "n_val_pairs": int(len(yva)),
                   "n_pos_total": int(y.sum())}, f, indent=2)
    print(f"* model saved ({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3])
