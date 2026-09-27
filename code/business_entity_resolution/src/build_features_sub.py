"""
Stage 3 (train split) - materialize pair features with NEGATIVE SUBSAMPLING.

The full train candidate matrix (321M pairs x 34 float32) would need ~43 GB,
which does not fit on disk.  Instead we keep:

  * EVERY candidate pair that is a ground-truth positive (from
    pos_s1.npy / pos_t.npy produced by src.eval_recall), and
  * a random sample of ~neg_per_s1 negatives per S1 row (default 3).

That yields ~12-13M pairs (~1.8 GB) - enough negatives for the ranker while
keeping every positive the model must learn to score high.

Usage:
  python -m src.build_features_sub <blob_dir> <cand_dir> <pos_dir> <out_dir> \
      <split> [workers] [neg_per_s1]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feats_fast import NF, BlobStore, compute_chunk  # noqa: E402

CH = 150_000          # pairs per worker job
BLOCK = 50_000        # S1 rows per sampling block


def run(blob_dir: str, cand_dir: str, pos_dir: str, out_dir: str, split: str,
        workers: int = 6, neg_per_s1: float = 3.0) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    flat = np.load(f"{cand_dir}/cand.flat.npy", mmap_mode="r")
    offs = np.load(f"{cand_dir}/cand.offs.npy", mmap_mode="r")
    score = np.load(f"{cand_dir}/cand.score.npy", mmap_mode="r")
    cnt = np.load(f"{cand_dir}/cand.count.npy", mmap_mode="r")
    n_s1 = len(offs) - 1
    n_pairs = len(flat)

    pos_s1 = np.load(f"{pos_dir}/pos_s1.npy")
    pos_t = np.load(f"{pos_dir}/pos_t.npy").astype(np.int64)
    pos_keys = np.unique((pos_s1 << np.int64(32)) | pos_t)
    del pos_s1, pos_t
    print(f"* {n_pairs:,} pairs / {n_s1:,} S1; {len(pos_keys):,} gt-positive "
          f"keys ({time.time()-t0:.0f}s)", file=sys.stderr)

    rng = np.random.default_rng(7)
    target_neg = neg_per_s1 * n_s1
    n_pos_kept = 0
    n_neg_kept = 0
    sel_s1, sel_t, sel_ksc, sel_kcn = [], [], [], []

    for lo in range(0, n_s1, BLOCK):
        hi = min(lo + BLOCK, n_s1)
        a, b = int(offs[lo]), int(offs[hi])
        if a == b:
            continue
        seg_t = np.asarray(flat[a:b])
        seg_s1 = np.repeat(np.arange(lo, hi, dtype=np.int64),
                           np.diff(np.asarray(offs[lo:hi + 1])))
        keys = (seg_s1 << np.int64(32)) | seg_t.astype(np.int64)
        if len(pos_keys):
            j = np.minimum(np.searchsorted(pos_keys, keys), len(pos_keys) - 1)
            is_pos = pos_keys[j] == keys
        else:
            is_pos = np.zeros(len(keys), dtype=bool)
        pos_idx = np.flatnonzero(is_pos)

        # adaptive negative keep-rate so the total lands near target_neg
        pos_frac = n_pos_kept / max(a, 1)
        remaining_neg = max(int((n_pairs - b) * (1.0 - pos_frac)), 1)
        keep = float(np.clip((target_neg - n_neg_kept) / remaining_neg,
                             1e-3, 0.05))
        r = rng.random(len(seg_t))
        neg_idx = np.flatnonzero(~is_pos & (r < keep))

        idx = np.concatenate([pos_idx, neg_idx])
        sel_s1.append(seg_s1[idx])
        sel_t.append(seg_t[idx])
        sel_ksc.append(np.asarray(score[a:b])[idx])
        sel_kcn.append(np.asarray(cnt[a:b])[idx])
        n_pos_kept += len(pos_idx)
        n_neg_kept += len(neg_idx)
        if (lo // BLOCK) % 10 == 0:
            print(f"  s1 {lo:,}/{n_s1:,}  pos={n_pos_kept:,} "
                  f"neg={n_neg_kept:,} keep={keep:.4f} ({time.time()-t0:.0f}s)",
                  file=sys.stderr)

    sel_s1 = np.concatenate(sel_s1)
    sel_t = np.concatenate(sel_t)
    sel_ksc = np.concatenate(sel_ksc)
    sel_kcn = np.concatenate(sel_kcn)
    n_sel = len(sel_s1)
    print(f"* selected {n_sel:,} pairs ({n_pos_kept:,} pos / "
          f"{n_neg_kept:,} neg) ({time.time()-t0:.0f}s)", file=sys.stderr)

    X = np.lib.format.open_memmap(f"{out_dir}/X.npy", mode="w+",
                                  dtype=np.float32, shape=(n_sel, NF))
    windows = [(a, min(a + CH, n_sel)) for a in range(0, n_sel, CH)]

    if workers <= 1:
        store = BlobStore(blob_dir, split)
        for wi, (a, b) in enumerate(windows):
            chunk = list(zip(sel_s1[a:b].tolist(), sel_t[a:b].tolist(),
                             sel_ksc[a:b].tolist(), sel_kcn[a:b].tolist()))
            X[a:b] = compute_chunk(store, chunk)
            if wi % 20 == 0:
                print(f"    {b:,}/{n_sel:,} ({time.time()-t0:.0f}s)",
                      file=sys.stderr)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        jobs = [(blob_dir, split, sel_s1[a:b].tolist(), sel_t[a:b].tolist(),
                 sel_ksc[a:b].tolist(), sel_kcn[a:b].tolist())
                for a, b in windows]
        with ctx.Pool(workers, initializer=_winit,
                      initargs=(blob_dir, split)) as pool:
            for wi, (a, b) in enumerate(windows):
                X[a:b] = pool.imap(_wjob, jobs[wi:wi + 1], chunksize=1).__next__()
                if wi % 20 == 0:
                    print(f"    {b:,}/{n_sel:,} ({time.time()-t0:.0f}s)",
                          file=sys.stderr)
    X.flush()
    np.save(f"{out_dir}/pair_s1.npy", sel_s1)
    np.save(f"{out_dir}/pair_t.npy", sel_t.astype(np.int64))
    with open(f"{out_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump({"n_pairs": int(n_sel), "n_pos": int(n_pos_kept),
                   "n_neg": int(n_neg_kept), "neg_per_s1": neg_per_s1,
                   "n_s1": int(n_s1), "n_pairs_total": int(n_pairs),
                   "split": split}, f, indent=2)
    print(f"* features written {X.shape} ({time.time()-t0:.0f}s)",
          file=sys.stderr)


def _winit(blob_dir: str, split: str) -> None:
    global _S
    _S = BlobStore(blob_dir, split)


def _wjob(args):
    blob_dir, split, i1s, ts, ksc, kcn = args
    store = _S if "_S" in globals() else BlobStore(blob_dir, split)
    return compute_chunk(store, list(zip(i1s, ts, ksc, kcn)))


if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2], a[3], a[4], a[5],
        workers=int(a[6]) if len(a) > 6 else 6,
        neg_per_s1=float(a[7]) if len(a) > 7 else 3.0)
