"""
Keep only the top-N highest-IDF-score candidates per S1 row.

The candidate builder already stores pairs in arbitrary within-group order;
this pass re-sorts each group by IDF score (descending) and keeps the first
top_n pairs.  On train data, top-100 keeps 99.39% of all ground-truth
positives (94.40% -> 93.82% absolute recall ceiling) while cutting the pair
count ~33%, which shortens LightGBM inference proportionally.

Outputs <out_dir>/cand.{flat,offs,score,count}.npy + config.json.

Usage:
  python -m src.truncate_cand <cand_dir_in> <cand_dir_out> <top_n>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

BLOCK_ROWS = 100_000


def run(cand_dir: str, out_dir: str, top_n: int) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    flat = np.load(f"{cand_dir}/cand.flat.npy", mmap_mode="r")
    offs = np.load(f"{cand_dir}/cand.offs.npy", mmap_mode="r")
    score = np.load(f"{cand_dir}/cand.score.npy", mmap_mode="r")
    cnt = np.load(f"{cand_dir}/cand.count.npy", mmap_mode="r")
    n_s1 = len(offs) - 1

    sel_parts = []
    counts = np.zeros(n_s1, dtype=np.int64)
    for lo in range(0, n_s1, BLOCK_ROWS):
        hi = min(lo + BLOCK_ROWS, n_s1)
        a, b = int(offs[lo]), int(offs[hi])
        if a == b:
            continue
        seg_t = np.asarray(flat[a:b])
        seg_sc = np.asarray(score[a:b])
        seg_cn = np.asarray(cnt[a:b])
        seg_s1 = np.repeat(np.arange(lo, hi, dtype=np.int64),
                           np.diff(np.asarray(offs[lo:hi + 1])))
        order = np.lexsort((-seg_sc, seg_s1))
        # keep first top_n within each group
        starts_g = np.repeat(offs[lo:hi].astype(np.int64),
                             np.diff(np.asarray(offs[lo:hi + 1]))) - a
        rank = np.arange(len(seg_t)) - starts_g[order]
        keep_sorted = rank < top_n
        sel = order[keep_sorted]
        sel_parts.append((seg_t[sel], seg_sc[sel], seg_cn[sel]))
        counts[lo:hi] = np.bincount(seg_s1[sel] - lo, minlength=hi - lo)
        if (lo // BLOCK_ROWS) % 5 == 0:
            print(f"  rows {lo:,}/{n_s1:,} kept={int(counts.sum()):,} "
                  f"({time.time()-t0:.0f}s)", file=sys.stderr)

    sel_t = np.concatenate([p[0] for p in sel_parts])
    sel_sc = np.concatenate([p[1] for p in sel_parts])
    sel_cn = np.concatenate([p[2] for p in sel_parts])
    del sel_parts
    new_offs = np.zeros(n_s1 + 1, dtype=np.int64)
    np.cumsum(counts, out=new_offs[1:])
    np.save(f"{out_dir}/cand.flat.npy", sel_t.astype(np.int32, copy=False))
    np.save(f"{out_dir}/cand.score.npy", sel_sc.astype(np.float32, copy=False))
    np.save(f"{out_dir}/cand.count.npy", sel_cn)
    np.save(f"{out_dir}/cand.offs.npy", new_offs)
    n_kept = int(new_offs[-1])
    cfg = {"top_n": top_n, "n_s1": int(n_s1), "n_pairs": n_kept,
           "source": cand_dir,
           "mean_cand": float(n_kept / max(n_s1, 1))}
    with open(f"{out_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"* kept {n_kept:,}/{len(flat):,} pairs "
          f"({100*n_kept/max(len(flat),1):.1f}%) in {time.time()-t0:.0f}s",
          file=sys.stderr)

if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2], int(a[3]))
