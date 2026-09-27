"""
Stage 2 (production) - vectorized candidate generation + IDF pre-ranking.

For every S1 record:
  1. select its k rarest blocking keys (soft df cap),
  2. pull all posting lists -> (s1, target) pairs with shared-key counts,
  3. score pairs with an IDF-weighted shared-key score (vectorized numpy),
  4. keep the top-M targets per S1 -> candidate set for the matcher.

Everything runs on flat int arrays in chunks, so multi-billion pair volumes
never materialize in RAM.

Outputs (per split): cand.flat.npy (int32 target idx), cand.offs.npy (int64),
cand.score.npy (float32 idf score), config.json.
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

CHUNK = 50_000          # S1 records per flush
TOP_M = 3000            # candidates kept per S1 after IDF ranking


def build_index(keys_dir: str, split: str):
    """CSR inverted index built from the two target sources (memory-lean)."""
    ids2, off2 = K.load_keys(keys_dir, split, 2)
    ids3, off3 = K.load_keys(keys_dir, split, 3)
    n2, n3 = len(off2) - 1, len(off3) - 1
    nt = n2 + n3
    v = int(max(ids2.max(), ids3.max())) + 1
    df = (np.bincount(ids2, minlength=v) + np.bincount(ids3, minlength=v)).astype(np.int32)
    # IMPORTANT: sort occurrences globally by key id.  Sorting the two sources
    # separately and concatenating is WRONG: the flat postings array would not
    # be globally ordered by key, so koff slices would misalign.
    ids_all = np.concatenate([ids2, ids3])
    order_key = np.argsort(ids_all, kind="stable")  # int64
    del ids_all
    # occurrence -> global target row (s2 rows 0..n2-1, s3 rows n2..n2+n3-1)
    recs2 = np.repeat(np.arange(n2, dtype=np.int64), np.diff(off2).astype(np.int64))
    recs3 = np.repeat(np.arange(n3, dtype=np.int64) + n2, np.diff(off3).astype(np.int64))
    recs = np.concatenate([recs2, recs3])
    del recs2, recs3
    order_rec = recs[order_key].astype(np.int32)
    del recs, order_key, ids2, ids3, off2, off3
    koff = np.zeros(v + 1, dtype=np.int64)
    np.cumsum(df, out=koff[1:])
    return {"df": df, "order": order_rec, "koff": koff, "V": v, "NT": nt}


def idf(df: np.ndarray, nt: int) -> np.ndarray:
    return np.log(1.0 + nt / (df + 1.0)).astype(np.float32)


def top_m_per_group(pairs: np.ndarray, scores: np.ndarray, counts: np.ndarray,
                    group_start: int, group_end: int, m: int):
    """Keep top-m (by score) pairs of group [group_start, group_end)."""
    seg = pairs[group_start:group_end]
    sc = scores[group_start:group_end]
    cn = counts[group_start:group_end]
    if len(seg) > m:
        # lexicographic tiebreak: score desc, count desc, target asc
        sel = np.lexsort((seg, -cn.astype(np.int64), -sc.astype(np.float64)))[:m]
        sel = np.sort(sel)
        return seg[sel], sc[sel], cn[sel]
    return seg, sc, cn


def run(keys_dir: str, out_dir: str, split: str, k: int = 6, thr: float = 2000.0,
        top_m: int = TOP_M) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    vocab = K.load_vocab(keys_dir)
    print("* building inverted index", file=sys.stderr)
    index = build_index(keys_dir, split)
    df = index["df"]
    w = idf(df, index["NT"])
    v = index["V"]

    # class mask without materializing a per-key string list (saves ~1.5GB):
    # key prefixes are 'cc|X...' where X is the class char; recover class by
    # re-deriving from the key-id -> string map in streaming fashion.
    cls = np.zeros(v, dtype=np.int8)
    for s, i in vocab.to_id.items():
        if i < v:
            ch = s.split("|", 1)[-1][0]
            # name/addr tokens, bigrams, and postal codes are admissible;
            # digit-signature keys (class D) are too brittle to block on.
            cls[i] = 0 if ch in ("n", "a", "B", "C", "z") else 1
    allowed = cls == 0

    ids, offs = K.load_keys(keys_dir, split, 1)
    n_s1 = len(offs) - 1
    print(f"* {n_s1:,} S1 records; k={k} thr={thr:g} top_m={top_m}", file=sys.stderr)

    flat_p, score_p, cnt_p = [], [], []
    offs_out = np.zeros(n_s1 + 1, dtype=np.int64)
    done = 0
    POST_BUDGET = 25_000_000     # max concatenated postings per flush
    start = 0
    while start < n_s1:
        end = min(start + CHUNK, n_s1)
        seg_pairs, seg_scores, seg_counts = [], [], []
        seg_off = np.zeros(end - start + 1, dtype=np.int64)
        budget = 0
        for i in range(start, end):
            ks = ids[offs[i]:offs[i + 1]]
            sel = B.select_keys(ks, df, k, thr, allowed)
            parts = []
            for kid in sel:
                a, b = index["koff"][kid], index["koff"][kid + 1]
                if b > a:
                    parts.append(index["order"][a:b])
                    budget += b - a
            if parts:
                cat = np.concatenate(parts)
                u, cnts = np.unique(cat, return_counts=True)
                sc = w[u] * cnts
                gs = seg_off[i - start]
                seg_pairs.append(u.astype(np.int32))
                seg_scores.append(sc)
                seg_counts.append(cnts.astype(np.int32))
                seg_off[i - start + 1] = gs + len(u)
            else:
                seg_off[i - start + 1] = seg_off[i - start]
            if budget > POST_BUDGET and i > start:
                end = i + 1
                break
        pairs_all = np.concatenate(seg_pairs) if seg_pairs else np.empty(0, np.int32)
        scores_all = np.concatenate(seg_scores) if seg_scores else np.empty(0, np.float32)
        counts_all = np.concatenate(seg_counts) if seg_counts else np.empty(0, np.int32)
        del seg_pairs, seg_scores, seg_counts

        keep_p, keep_s, keep_c = [], [], []
        pos = 0
        for gi in range(end - start):
            g0 = pos
            g1 = pos + int(seg_off[gi + 1] - seg_off[gi])
            pos = g1
            if g1 > g0:
                p, s, c = top_m_per_group(pairs_all, scores_all, counts_all,
                                          g0, g1, top_m)
                keep_p.append(p)
                keep_s.append(s)
                keep_c.append(c)
                offs_out[start + gi + 1] = offs_out[start + gi] + len(p)
            else:
                offs_out[start + gi + 1] = offs_out[start + gi]
        if keep_p:
            flat_p.append(np.concatenate(keep_p))
            score_p.append(np.concatenate(keep_s))
            cnt_p.append(np.concatenate(keep_c))
        start = end
        print(f"    {start:,}/{n_s1:,} ({time.time()-t0:.0f}s)", file=sys.stderr)

    flat = np.concatenate(flat_p) if flat_p else np.empty(0, np.int32)
    score = np.concatenate(score_p) if score_p else np.empty(0, np.float32)
    cnt = np.concatenate(cnt_p).astype(np.int16) if cnt_p else np.empty(0, np.int16)
    n_pairs = len(flat)
    mean_cand = float(n_pairs / max(n_s1, 1))
    del flat_p, score_p, cnt_p

    def _atomic_save(path, arr):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        print(f"    wrote {os.path.basename(path)} "
              f"({os.path.getsize(path)/1e9:.2f} GB)", file=sys.stderr)

    _atomic_save(f"{out_dir}/cand.offs.npy", offs_out)
    _atomic_save(f"{out_dir}/cand.flat.npy", flat.astype(np.int32, copy=False))
    del flat
    _atomic_save(f"{out_dir}/cand.score.npy", score.astype(np.float32, copy=False))
    del score
    _atomic_save(f"{out_dir}/cand.count.npy", cnt)
    cfg = {"k": k, "thr": thr, "top_m": top_m, "split": split, "n_s1": int(n_s1),
           "n_pairs": n_pairs,
           "mean_cand": mean_cand}
    with open(f"{out_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"* wrote {n_pairs:,} pairs ({mean_cand:.1f}/S1) "
          f"in {time.time()-t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2], a[3],
        k=int(a[4]) if len(a) > 4 else 6,
        thr=float(a[5]) if len(a) > 5 else 2000.0,
        top_m=int(a[6]) if len(a) > 6 else TOP_M)
