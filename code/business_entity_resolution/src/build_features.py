"""
Stage 3 driver - materialize pair features from candidates (memory-bounded).

Streams candidate pairs from memmaps, computes features via BlobStore +
compute_chunk, writes X as a float32 memmap.  Chunk boundaries respect the
per-group (S1) boundaries so multiprocessing workers never split a group.

Usage:
  python -m src.build_features <blob_dir> <cand_dir> <out_dir> <split> [workers]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feats_fast import NF, BlobStore, compute_chunk  # noqa: E402

CH = 150_000


def _chunks_by_group(offs, n_pairs):
    """Yield (start_pair, end_pair) windows that never split an S1 group."""
    start = 0
    while start < n_pairs:
        target = min(start + CH, n_pairs)
        # extend to group boundary
        end = int(np.searchsorted(offs, target, side="right") - 1)
        end_pair = int(offs[end]) if end < len(offs) - 1 else n_pairs
        if end_pair <= start:            # single group bigger than CH
            end = end + 1
            end_pair = int(offs[end]) if end < len(offs) else n_pairs
        yield start, end_pair
        start = end_pair


def _winit(blob_dir: str, split: str) -> None:
    global _S
    _S = BlobStore(blob_dir, split)


def _wchunk(args):
    blob_dir, split, i1s, ts, ksc, kcn = args
    store = _S if "_S" in globals() else BlobStore(blob_dir, split)
    return compute_chunk(store, list(zip(i1s, ts, ksc, kcn)))


def run(blob_dir: str, cand_dir: str, out_dir: str, split: str,
        workers: int = 1) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    flat = np.load(f"{cand_dir}/cand.flat.npy", mmap_mode="r")
    offs = np.load(f"{cand_dir}/cand.offs.npy", mmap_mode="r")
    score = np.load(f"{cand_dir}/cand.score.npy", mmap_mode="r")
    cnt = np.load(f"{cand_dir}/cand.count.npy", mmap_mode="r")
    n_s1 = len(offs) - 1
    n_pairs = len(flat)
    print(f"* {n_pairs:,} pairs over {n_s1:,} S1 -> {split}", file=sys.stderr)

    X = np.lib.format.open_memmap(f"{out_dir}/X.npy", mode="w+",
                                  dtype=np.float32, shape=(n_pairs, NF))
    pair_s1 = np.lib.format.open_memmap(f"{out_dir}/pair_s1.npy", mode="w+",
                                        dtype=np.int64, shape=(n_pairs,))
    pair_t = np.lib.format.open_memmap(f"{out_dir}/pair_t.npy", mode="w+",
                                       dtype=np.int64, shape=(n_pairs,))

    windows = list(_chunks_by_group(offs, n_pairs))
    print(f"  {len(windows)} windows", file=sys.stderr)

    if workers <= 1:
        for wi, (a, b) in enumerate(windows):
            i1s = np.searchsorted(offs, flat[a:b], side='right') - 1
            chunk = list(zip(i1s.tolist(), flat[a:b].tolist(),
                             score[a:b].tolist(), cnt[a:b].tolist()))
            X[a:b] = compute_chunk(BlobStore(blob_dir, split), chunk)
            pair_s1[a:b] = i1s
            pair_t[a:b] = flat[a:b]
            if wi % 20 == 0:
                print(f"    {b:,}/{n_pairs:,} ({time.time()-t0:.0f}s)",
                      file=sys.stderr)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        jobs = []
        offs_full = np.load(f"{cand_dir}/cand.offs.npy", mmap_mode="r")
        for (a, b) in windows:
            i1s = np.searchsorted(offs_full, flat[a:b], side='right') - 1
            jobs.append((blob_dir, split, i1s.tolist(), flat[a:b].tolist(),
                         score[a:b].tolist(), cnt[a:b].tolist()))
        with ctx.Pool(workers, initializer=_winit,
                      initargs=(blob_dir, split)) as pool:
            res_iter = pool.imap(_wchunk, jobs)
            done = 0
            for (a, b), res in zip(windows, res_iter):
                X[a:b] = res
                i1s = np.searchsorted(offs, flat[a:b], side='right') - 1
                pair_s1[a:b] = i1s
                pair_t[a:b] = flat[a:b]
                done += 1
                if done % 20 == 0:
                    print(f"    {done}/{len(windows)} windows "
                          f"({time.time()-t0:.0f}s)", file=sys.stderr)
    X.flush()
    pair_s1.flush()
    pair_t.flush()
    print(f"* features written {X.shape} ({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2], a[3], a[4], workers=int(a[5]) if len(a) > 5 else 1)