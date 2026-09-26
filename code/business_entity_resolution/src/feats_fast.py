"""
Pair feature computation from blobs (fast path).

A worker receives a chunk of (s1_row, target_idx, key_score, key_count) tuples,
reads text slices from memory-mapped blobs, computes features with rapidfuzz,
and returns a float32 matrix chunk.  Designed for multiprocessing over chunks.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import normalize as N  # noqa: E402
from src.features import NF  # noqa: E402


class BlobStore:
    def __init__(self, blob_dir: str, split: str) -> None:
        self.split = split
        self.name = {}
        self.addr = {}
        self.dig = {}
        self.country = {}
        self.postal = {}
        self.nkeys = {}
        for s in (1, 2, 3):
            self.name[s] = _MM(f"{blob_dir}/{split}_s{s}.name.bin")
            self.addr[s] = _MM(f"{blob_dir}/{split}_s{s}.addr.bin")
            self.dig[s] = _MM(f"{blob_dir}/{split}_s{s}.dig.bin")
            self.country[s] = np.load(f"{blob_dir}/{split}_s{s}.country.npy")
            self.postal[s] = np.load(f"{blob_dir}/{split}_s{s}.postal.npy")
            self.nkeys[s] = np.load(f"{blob_dir}/{split}_s{s}.nkeys.npy")
        self._name_off = {s: np.load(f"{blob_dir}/{split}_s{s}.name.off.npy")
                          for s in (1, 2, 3)}
        self._addr_off = {s: np.load(f"{blob_dir}/{split}_s{s}.addr.off.npy")
                          for s in (1, 2, 3)}
        self._dig_off = {s: np.load(f"{blob_dir}/{split}_s{s}.dig.off.npy")
                         for s in (1, 2, 3)}
        self._len = {s: len(self.country[s]) for s in (1, 2, 3)}
        self.n2 = self._len[2]

    def rec(self, s: int, i: int):
        no, noff = self.name[s], self._name_off[s]
        ao, aoff = self.addr[s], self._addr_off[s]
        do, doff = self.dig[s], self._dig_off[s]
        name = no[noff[i]:noff[i + 1]].decode("utf-8")
        addr = ao[aoff[i]:aoff[i + 1]].decode("utf-8")
        dig = do[doff[i]:doff[i + 1]].decode("utf-8")
        return name, addr, dig, self.country[s][i], self.postal[s][i]


class _MM:
    """Minimal mmap wrapper (bytes-like, sliceable)."""

    def __init__(self, path: str) -> None:
        import mmap
        f = open(path, "rb")
        self._m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._f = f

    def __getitem__(self, sl):
        return self._m[sl]

    def __len__(self):
        return len(self._m)


def _tokens_cached_factory():
    cache = {}

    def get(kind, text):
        key = (kind, text)
        v = cache.get(key)
        if v is None:
            if kind == "n":
                v = frozenset(N.name_tokens(text))
            else:
                v = frozenset(N.addr_tokens(text))
            if len(cache) > 300_000:
                cache.clear()
            cache[key] = v
        return v

    return get


def _charbigram_cos(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ga = Counter(a[i:i+2] for i in range(len(a) - 1))
    gb = Counter(b[i:i+2] for i in range(len(b) - 1))
    dot = sum((ga & gb).values())
    na, nb = sum(ga.values()), sum(gb.values())
    return dot / (na ** 0.5 * nb ** 0.5) if na and nb else 0.0


def _legal_set(toks):
    return {t for t in toks if t in N.LEGAL_TOKENS}


def compute_chunk(store: BlobStore, chunk):
    """chunk: iterable of (s1_row, target_idx, key_score, key_count)."""
    tok = _tokens_cached_factory()
    out = np.empty((len(chunk), NF), dtype=np.float32)
    n2 = store.n2
    for r, (i1, t, ksc, kcn) in enumerate(chunk):
        n1, a1, d1, c1, p1 = store.rec(1, i1)
        if t < n2:
            s = 2
            i2 = t
        else:
            s = 3
            i2 = t - n2
        n2t, a2, d2, c2, p2 = store.rec(s, i2)

        t1 = tok("n", n1)
        t2 = tok("n", n2)
        at1 = tok("a", a1)
        at2 = tok("a", a2)

        # ---- name ----
        jac = N.jaccard(t1, t2)
        ov = N.overlap_coef(t1, t2)
        suf1, suf2 = _legal_set(t1), _legal_set(t2)
        core1, core2 = t1 - suf1, t2 - suf2
        core_jac = N.jaccard(core1, core2) if (core1 or core2) else jac
        dset1, dset2 = set(d1.split("|")) - {""}, set(d2.split("|")) - {""}
        p1set = {x for x in dset1 if len(x) in (5, 6) and x.isdigit()}
        p2set = {x for x in dset2 if len(x) in (5, 6) and x.isdigit()}
        st1 = {x for x in t1 if x in N.STATE_TOKENS and len(x) == 2}
        st2 = {x for x in t2 if x in N.STATE_TOKENS and len(x) == 2}
        e1, e2 = (not a1.strip()), (not a2.strip())
        both_empty = e1 and e2
        any_empty = e1 or e2
        _z = 1.0 if both_empty else (0.0 if any_empty else None)

        def addr_ratio(fn):
            return 1.0 if both_empty else (0.0 if any_empty else fn)

        out[r] = [
            # name (13)
            jac, ov,
            fuzz.ratio(n1, n2) / 100.0,
            fuzz.partial_ratio(n1, n2) / 100.0,
            fuzz.token_set_ratio(n1, n2) / 100.0,
            fuzz.token_sort_ratio(n1, n2) / 100.0,
            _charbigram_cos(n1, n2),
            JaroWinkler.similarity(n1, n2),
            min(len(n1), len(n2)) / max(len(n1), len(n2), 1),
            (N.jaccard(suf1, suf2) if (suf1 or suf2) else 1.0),
            1.0 if (suf1 and suf2) else 0.0,
            1.0 if (not t1 and not t2) else 0.0,
            core_jac,
            # address (16)
            N.jaccard(at1, at2),
            N.overlap_coef(at1, at2),
            addr_ratio(fuzz.ratio(a1, a2) / 100.0),
            addr_ratio(fuzz.partial_ratio(a1, a2) / 100.0),
            addr_ratio(fuzz.token_set_ratio(a1, a2) / 100.0),
            addr_ratio(fuzz.token_sort_ratio(a1, a2) / 100.0),
            _charbigram_cos(a1, a2) if not both_empty else 0.0,
            addr_ratio(JaroWinkler.similarity(a1, a2)),
            min(len(a1), len(a2)) / max(len(a1), len(a2), 1),
            N.jaccard(dset1, dset2),
            1.0 if (dset1 and dset2 and (dset1 <= dset2 or dset2 <= dset1)) else 0.0,
            1.0 if (p1set and p1set & p2set) else 0.0,
            1.0 if (st1 and st1 & st2) else 0.0,
            1.0 if (st1 != st2) else 0.0,
            1.0 if (e1 != e2) else 0.0,
            1.0 if both_empty else 0.0,
            # meta (5)
            1.0 if c1 == c2 else 0.0,
            float(ksc),
            float(kcn),
            float(kcn) / max(int(store.nkeys[1][i1]),
                             int(store.nkeys[s][i2]), 1),
            float(p1 != 0 and p2 != 0 and p1 == p2),
        ]
    return out
