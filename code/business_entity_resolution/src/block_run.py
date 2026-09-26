"""
Stage 2 - candidate generation.

Builds the inverted index over the target (S2+S3) blocking keys once, saves it
to disk, and materializes candidate sets for every S1 record of a split as:

* ``cand.flat.npy``  int32  concatenated candidate record indices
* ``cand.offs.npy``  int64  per-S1 offsets into the flat array

Run:  python -m src.block_run <cache_dir> <keys_dir> <out_dir> <split> <k> <thr>
e.g.  python -m src.block_run work/cache work/keys work/cand_train train 6 2000
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import blocking as B  # noqa: E402
from src import keys as K  # noqa: E402


def run(cache_dir: str, keys_dir: str, out_dir: str, split: str,
        k: int = 6, thr: float = 2000.0) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    idx_dir = f"{out_dir}/index"
    vocab = K.load_vocab(keys_dir)

    index_path = f"{idx_dir}/order.npy"
    if os.path.exists(index_path):
        print("* loading saved inverted index")
        index = B.load_index(idx_dir)
    else:
        print("* building inverted index")
        index = B.load_target_index(keys_dir, split)
        B.save_index(index, idx_dir)

    cls = B.class_array(vocab, index["V"])
    # Name + address tokens only; the digit-signature key is too brittle and
    # zips are usually covered by the zip key itself.  'thr' caps boilerplate.
    allowed = np.isin(cls, [B.NAME, B.ADDR])

    ids, offs = K.load_keys(keys_dir, split, 1)
    n_s1 = len(offs) - 1
    print(f"* generating candidates for {n_s1:,} S1 records "
          f"(k={k}, thr={thr:g}) ({time.time()-t0:.0f}s)")

    flat_parts: list[np.ndarray] = []
    offs_out = np.zeros(n_s1 + 1, dtype=np.int64)
    CH = 200_000
    for start in range(0, n_s1, CH):
        end = min(start + CH, n_s1)
        for i in range(start, end):
            ks = ids[offs[i]:offs[i + 1]]
            c = B.candidates_for(ks, index, k, thr, allowed)
            flat_parts.append(c)
            offs_out[i + 1] = offs_out[i] + len(c)
        print(f"    {end:,}/{n_s1:,} ({time.time()-t0:.0f}s)", file=sys.stderr)

    flat = np.concatenate(flat_parts) if flat_parts else np.empty(0, dtype=np.int32)
    np.save(f"{out_dir}/cand.flat.npy", flat)
    np.save(f"{out_dir}/cand.offs.npy", offs_out)
    cfg = {"k": k, "thr": thr, "split": split,
           "n_s1": int(n_s1), "n_pairs": int(len(flat)),
           "mean_cand": float(len(flat) / max(n_s1, 1))}
    with open(f"{out_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"* wrote {out_dir}: {len(flat):,} candidate pairs "
          f"({len(flat)/max(n_s1,1):.1f}/S1) total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2], a[3], a[4],
        k=int(a[5]) if len(a) > 5 else 6,
        thr=float(a[6]) if len(a) > 6 else 2000.0)
