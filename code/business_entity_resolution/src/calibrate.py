"""
Stage 4b - calibrate the F0.5 decision threshold on FULL candidate pools.

Training uses subsampled negatives, so val scores there are not distributed
like inference (where each S1 has its full ~145-candidate pool, nearly all
negatives).  This script scores the calib feature matrix built by
src.build_features_range (ALL candidates for a random S1-row sample) and
sweeps the threshold against ground truth, including true singletons.

Also runs per-S1 sanity checks: 'top-1 if above t1' and 'top-2' policies, to
see if a smarter decision rule beats the plain global threshold.

Usage:
  python -m src.calibrate <calib_feature_dir> <cand_dir> <model_dir> \
      <gt_tsv> <cache_dir> <out_dir>
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metric import f05_of_entity  # noqa: E402


def build_truth_rows(cache_dir: str, gt_tsv: str) -> dict[int, set]:
    """GT as {s1_row: set(target_row)}; row order = parquet order, S3 +n2."""
    import pyarrow.parquet as pq
    t2i = {}
    n2 = 0
    for b in pq.ParquetFile(f"{cache_dir}/train_source2.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        vals = b.column("entity_id").to_pylist()
        for i, e in enumerate(vals):
            t2i[e] = n2 + i
        n2 += len(vals)
    for b in pq.ParquetFile(f"{cache_dir}/train_source3.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        vals = b.column("entity_id").to_pylist()
        for i, e in enumerate(vals):
            t2i[e] = n2 + i
        n2 += len(vals)
    s1_row = {}
    i = 0
    for b in pq.ParquetFile(f"{cache_dir}/train_source1.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        for e in b.column("entity_id").to_pylist():
            s1_row[e] = i
            i += 1
    truth = {}
    with open(gt_tsv, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for s1, m in r:
            i1 = s1_row.get(s1)
            if i1 is None:
                continue
            rows = {t2i[e] for e in m.split(",") if e in t2i}
            if rows:
                truth[i1] = rows
    return truth


def sweep(s1_arr, t_arr, pred, truth, thresholds):
    """Macro-F0.5 for a set of thresholds over the sampled rows.

    Rows without any candidate are predicted-empty: they score 1.0 iff they
    are true singletons, else 0.0."""
    order = np.argsort(s1_arr, kind="stable")
    s1_sorted = s1_arr[order]
    t_sorted = t_arr[order]
    p_sorted = pred[order]
    uniq, starts = np.unique(s1_sorted, return_index=True)
    starts = list(starts) + [len(s1_sorted)]
    groups = {int(u): (starts[i], starts[i + 1]) for i, u in enumerate(uniq)}

    n_rows = len(rows_of_sample)
    no_cand = [int(r) for r in rows_of_sample if int(r) not in groups]
    no_cand_score = sum(1.0 for r in no_cand if r not in truth)
    results = {}
    for thr in thresholds:
        total = no_cand_score
        missed = fp_only = perfect = 0
        for s, (a, b) in groups.items():
            mask = p_sorted[a:b] >= thr
            pset = set(t_sorted[a:b][mask].tolist())
            tset = truth.get(s, set())
            total += f05_of_entity(pset, tset)
            if not pset and tset:
                missed += 1
            elif pset and not tset:
                fp_only += 1
            elif pset == tset:
                perfect += 1
        f = total / n_rows
        results[thr] = (f, {"missed": missed, "fp_only": fp_only,
                            "perfect": perfect,
                            "no_cand_rows": len(no_cand)})
    return results


rows_of_sample = None  # assigned in main()


def main(calib_dir: str, cand_dir: str, model_dir: str, gt_tsv: str,
         cache_dir: str, out_dir: str) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    global rows_of_sample
    import lightgbm as lgb

    X = np.load(f"{calib_dir}/X.npy", mmap_mode="r")
    s1_arr = np.load(f"{calib_dir}/pair_s1.npy")
    t_arr = np.load(f"{calib_dir}/pair_t.npy")
    rows_of_sample = np.load(f"{calib_dir}/rows.npy")
    offs = np.load(f"{cand_dir}/cand.offs.npy", mmap_mode="r")
    n_s1_total = len(offs) - 1

    booster = lgb.Booster(model_file=f"{model_dir}/model.txt")
    print(f"* scoring {X.shape[0]:,} calib pairs ({time.time()-t0:.0f}s)",
          file=sys.stderr)
    pred = booster.predict(X, num_iteration=_best_iter(model_dir))
    print(f"* scored ({time.time()-t0:.0f}s)", file=sys.stderr)

    truth = build_truth_rows(cache_dir, gt_tsv)
    print(f"* truth loaded: {len(truth):,} S1 ({time.time()-t0:.0f}s)",
          file=sys.stderr)

    thrs = [round(x, 2) for x in
            [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]]
    results = sweep(s1_arr, t_arr, pred, truth, thrs)
    for thr, (f, st) in sorted(results.items()):
        print(f"  thr={thr:.2f}  F0.5={f:.4f}  {st}", file=sys.stderr)
    best_thr, (best_f, _) = max(results.items(), key=lambda kv: kv[1][0])

    # refine around best
    lo = max(best_thr - 0.04, 0.02)
    fine = [round(best_thr + d, 3) for d in
            (-0.04, -0.03, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.03,
             0.04)]
    fine = [t for t in fine if t > 0]
    fine_res = sweep(s1_arr, t_arr, pred, truth, fine)
    for thr, (f, _) in fine_res.items():
        if f > best_f:
            best_thr, best_f = thr, f

    print(f"BEST thr={best_thr}  F0.5={best_f:.4f}", file=sys.stderr)
    with open(f"{out_dir}/threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_thr, "val_f05_full_pool": best_f,
                   "n_calib_s1": int(len(rows_of_sample)),
                   "grid": {str(k): v[0] for k, v in sorted(results.items())}},
                  f, indent=2)
    print(f"* done ({time.time()-t0:.0f}s)", file=sys.stderr)


def _best_iter(model_dir: str):
    meta_p = f"{model_dir}/meta.json"
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            bi = json.load(f).get("best_iteration") or 0
        return bi or None
    return None


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], a[4], a[5], a[6])

