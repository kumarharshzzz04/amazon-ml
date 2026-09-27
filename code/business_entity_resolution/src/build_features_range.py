"""
Calibration feature builder - ALL candidate pairs for a random sample of S1
rows.  Used to pick the F0.5 threshold on a score distribution that matches
inference (full ~145-candidate pools), unlike the negative-subsampled training
matrix from build_features_sub.

Saves: X.npy, pair_s1.npy, pair_t.npy, rows.npy (the sampled S1 rows).

Usage:
  python -m src.build_features_range <blob_dir> <cand_dir> <out_dir> <split> \
      <n_rows> [seed] [workers]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feats_fast import NF, BlobStore, compute_chunk  # noqa: E402

CH = 150_000


def run(blob_dir: str, cand_dir: str, out_dir: str, split: str,
        n_rows: int, seed: int = 13, workers: int = 6) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    flat = np.load(f"{cand_dir}/cand.flat.npy", mmap_mode="r")
    offs = np.load(f"{cand_dir}/cand.offs.npy", mmap_mode="r")
    score = np.load(f"{cand_dir}/cand.score.npy", mmap_mode="r")
    cnt = np.load(f"{cand_dir}/cand.count.npy", mmap_mode="r")
    n_s1 = len(offs) - 1

    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n_s1, size=min(n_rows, n_s1), replace=False))
    np.save(f"{out_dir}/rows.npy", rows)

    sel_s1, sel_t, sel_ksc, sel_kcn = [], [], [], []
    for lo in range(0, len(rows), 5_000):
        rr = rows[lo:lo + 5_000]
        a, b = int(offs[rr[0]]), int(offs[rr[-1] + 1])
        seg_t = np.asarray(flat[a:b])
        seg_s1 = np.repeat(rr, np.diff(np.asarray(offs[rr[0]:rr[-1] + 2])))
        sel_s1.append(seg_s1)
        sel_t.append(seg_t)
        sel_ksc.append(np.asarray(score[a:b]))
        sel_kcn.append(np.asarray(cnt[a:b]))
    sel_s1 = np.concatenate(sel_s1)
    sel_t = np.concatenate(sel_t)
    sel_ksc = np.concatenate(sel_ksc)
    sel_kcn = np.concatenate(sel_kcn)
    n_sel = len(sel_s1)
    print(f"* calib: {len(rows):,} S1 rows -> {n_sel:,} pairs "
          f"({time.time()-t0:.0f}s)", file=sys.stderr)

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
            for wi, res in enumerate(pool.imap(_wjob, jobs, chunksize=1)):
                a, b = windows[wi]
                X[a:b] = res
                if wi % 20 == 0:
                    print(f"    {b:,}/{n_sel:,} ({time.time()-t0:.0f}s)",
                          file=sys.stderr)
    X.flush()
    np.save(f"{out_dir}/pair_s1.npy", sel_s1)
    np.save(f"{out_dir}/pair_t.npy", sel_t.astype(np.int64))
    print(f"* calib features written {X.shape} ({time.time()-t0:.0f}s)",
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
    run(a[1], a[2], a[3], a[4], int(a[5]),
        seed=int(a[6]) if len(a) > 6 else 13,
        workers=int(a[7]) if len(a) > 7 else 6)
