"""
Blocking stage: inverted index over blocking keys + candidate generation.

A target record (S2/S3) is a candidate for a source record (S1) when they share
at least one *selected* blocking key.  Selection keeps only each record's ``k``
rarest keys (by document frequency), optionally restricted to key classes, with
high-frequency boilerplate keys excluded.  That keeps the candidate volume
manageable while preserving recall.

The inverted index is built once per target set and persisted:

* ``df.npy``       int32[V]  document frequency of every key id
* ``order.npy``    int32[P]  target-record index of the p-th (key, record) pair,
                             sorted by key id (CSR over key ids)
* ``koff.npy``     int64[V+1] offsets into ``order`` for each key id

Only int32/int64 arrays are used; the whole index for 10.3M target records and
~122M postings fits comfortably in RAM.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import keys as K  # noqa: E402

NAME, ADDR, DIGITS, ZIP = 0, 1, 2, 3
CLASS_OF_PREFIX = {"n": NAME, "a": ADDR, "D": DIGITS, "z": ZIP}


def load_target_index(keys_dir: str, split: str) -> dict:
    """Load the two target sources and build the CSR inverted index."""
    t0 = time.time()
    ids2, off2 = K.load_keys(keys_dir, split, 2)
    ids3, off3 = K.load_keys(keys_dir, split, 3)
    n2, n3 = len(off2) - 1, len(off3) - 1
    NT = n2 + n3

    # Concatenate key arrays; target record index = position in concat order.
    ids_t = np.concatenate([ids2, ids3])
    recs_per = np.diff(np.concatenate([off2[:-1], off3[:-1] + off2[-1], [len(ids_t)]]))
    del ids2, ids3

    V = int(ids_t.max()) + 1
    df = np.bincount(ids_t, minlength=V).astype(np.int32)
    # CSR: sort postings by key id.
    order_key = np.argsort(ids_t, kind="stable").astype(np.int32)  # posting -> slot
    # Map each posting to its record index, then reorder.
    occ_rec = np.repeat(np.arange(NT, dtype=np.int32), recs_per.astype(np.int64))
    order_rec = occ_rec[order_key]
    del occ_rec, order_key, ids_t

    koff = np.zeros(V + 1, dtype=np.int64)
    np.cumsum(df, out=koff[1:])

    print(f"  index: V={V:,} postings={len(order_rec):,} targets={NT:,} "
          f"({time.time()-t0:.0f}s)", file=sys.stderr)
    return {"df": df, "order": order_rec, "koff": koff, "V": V, "NT": NT,
            "n2": n2, "n3": n3}


def save_index(index: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/df.npy", index["df"])
    np.save(f"{out_dir}/order.npy", index["order"])
    np.save(f"{out_dir}/koff.npy", index["koff"])
    meta = {"V": index["V"], "NT": index["NT"], "n2": index["n2"], "n3": index["n3"]}
    with open(f"{out_dir}/meta.json", "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f)


def load_index(in_dir: str) -> dict:
    import json
    with open(f"{in_dir}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return {"df": np.load(f"{in_dir}/df.npy"),
            "order": np.load(f"{in_dir}/order.npy"),
            "koff": np.load(f"{in_dir}/koff.npy"), **meta}


def class_array(vocab: K.KeyVocab, V: int) -> np.ndarray:
    """Key-class per key id (NAME/ADDR/DIGITS/ZIP)."""
    cls = np.full(V, ADDR, dtype=np.int8)
    for s, i in vocab.to_id.items():
        if i < V:
            cls[i] = CLASS_OF_PREFIX.get(s[0], ADDR)
    return cls


def select_keys(keys: np.ndarray, df: np.ndarray, k: int, thr: float,
                allowed: np.ndarray | None, hard_cap: float = np.inf) -> np.ndarray:
    """Return the <=k rarest admissible key ids of one record.

    ``thr`` is a *soft* cap applied only when the record has more than ``k``
    admissible keys under it; key-poor records fall back to their rarest keys
    whatever their document frequency (up to ``hard_cap``).  This matters:
    thresholding first empties the selection exactly for the generic-looking
    records whose only shared keys are mid-frequency.
    """
    vals = df[keys]
    if allowed is not None:
        m = allowed[keys]
        keys, vals = keys[m], vals[m]
    if len(keys) == 0:
        return keys
    order = np.argsort(vals, kind="stable")
    keys, vals = keys[order], vals[order]
    if thr < np.inf and len(keys) > k and vals[k - 1] > thr:
        # more than k keys but the k-th rarest already exceeds the soft cap:
        # keep only keys under the cap when that still leaves >= k keys.
        under = vals <= thr
        if under.sum() >= k:
            keys = keys[under]
    if hard_cap < np.inf:
        keys = keys[vals <= hard_cap]
    return keys[:k]


def candidates_for(s1_keys: np.ndarray, index: dict, k: int, thr: float,
                   allowed: np.ndarray | None, hard_cap: float = np.inf,
                   min_shared: int = 1) -> np.ndarray:
    """Deduplicated candidate record indices for one S1 record.

    ``min_shared`` requires a target to share at least that many of the
    selected keys (co-occurrence blocking), which slashes volume at a small
    recall cost.
    """
    ks = select_keys(s1_keys, index["df"], k, thr, allowed, hard_cap)
    parts = []
    for kid in ks:
        a, b = index["koff"][kid], index["koff"][kid + 1]
        if b > a:
            parts.append(index["order"][a:b])
    if not parts:
        return np.empty(0, dtype=np.int32)
    if len(parts) == 1:
        return parts[0] if min_shared <= 1 else np.empty(0, dtype=np.int32)
    cat = np.concatenate(parts)
    if min_shared <= 1:
        return np.unique(cat)
    uniq, cnts = np.unique(cat, return_counts=True)
    return uniq[cnts >= min_shared]
