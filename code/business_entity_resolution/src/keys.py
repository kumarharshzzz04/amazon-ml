"""
Blocking-key extraction and caching.

Every record is reduced to a small set of *blocking keys*:

  ``n<tok>``  a canonicalized business-name token        (e.g. ``ncapital``)
  ``a<tok>``  a canonicalized address token              (e.g. ``a108``, ``arome``)
  ``D<digits>`` the sorted set of address digit runs     (order-insensitive)
  ``z<digits>`` a 5/6-digit postal code (US ZIP / India PIN / FR code postal)

Keys are interned into a global vocabulary of integer ids and stored per source
as a flat ``int32`` array plus an offset array (CSR layout).  Document
frequencies then come from ``np.bincount`` and the inverted index from a single
``argsort``, both of which are effectively instant compared with re-tokenizing
15M records for every experiment.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import normalize as N  # noqa: E402


def record_keys(name: str, addr: str, country: str = "") -> set[str]:
    """All blocking keys of one record (accepts raw or normalized text).

    Keys are prefixed with the country: matches never cross countries, so
    partitioning halves (or better) every posting list for free.
    """
    c = (country or "").strip().lower()
    pre = c + "|" if c else ""
    keys = set()
    ntoks = N.name_tokens(name)
    for t in ntoks:
        keys.add(pre + "n" + t)
    atoks = N.addr_tokens(addr)
    for t in atoks:
        keys.add(pre + "a" + t)
    # sorted adjacent token bigrams: robust to word-order transposition
    for toks, tag in ((ntoks, "B"), (atoks, "C")):
        for x, y in zip(toks, toks[1:]):
            a, b = (x, y) if x <= y else (y, x)
            keys.add(pre + tag + a + "+" + b)
    digs = N.numeric_tokens(addr)
    if digs:
        uniq = sorted(set(digs))
        keys.add(pre + "D" + "|".join(uniq))
        for d in uniq:
            if len(d) in (5, 6):
                keys.add(pre + "z" + d)
    return keys


class KeyVocab:
    """String key -> int id, built incrementally."""

    def __init__(self) -> None:
        self.to_id: dict[str, int] = {}

    def intern(self, key: str) -> int:
        i = self.to_id.get(key)
        if i is None:
            i = len(self.to_id)
            self.to_id[key] = i
        return i

    def __len__(self) -> int:
        return len(self.to_id)


def encode_source(parquet_path: str, vocab: KeyVocab) -> tuple[np.ndarray, np.ndarray]:
    """Encode one source Parquet file into (flat_ids, offsets)."""
    import pyarrow.parquet as pq

    ids: list[int] = []
    offsets: list[int] = [0]
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    seen = 0
    t0 = time.time()
    for b in pf.iter_batches(batch_size=200_000, columns=["entity_id", "name_norm", "addr_norm", "country"]):
        names = b.column("name_norm").to_pylist()
        addrs = b.column("addr_norm").to_pylist()
        countries = b.column("country").to_pylist()
        eids = b.column("entity_id").to_pylist()
        for e, n, a, c in zip(eids, names, addrs, countries):
            for k in record_keys(n, a, c):
                ids.append(vocab.intern(k))
            offsets.append(len(ids))
        seen += len(names)
        print(f"    {os.path.basename(parquet_path)} {seen:,}/{total:,} "
              f"({time.time()-t0:.0f}s)", file=sys.stderr)
    return np.asarray(ids, dtype=np.int32), np.asarray(offsets, dtype=np.int64)


def build_cache(cache_dir: str, out_dir: str, splits=("train", "test")) -> None:
    """Build (flat_ids, offsets) for every source and persist them + the vocab."""
    os.makedirs(out_dir, exist_ok=True)
    vocab = KeyVocab()
    for split in splits:
        for s in (1, 2, 3):
            src = f"{cache_dir}/{split}_source{s}.parquet"
            out = f"{out_dir}/{split}_source{s}"
            print(f"* encoding {src}")
            ids, offs = encode_source(src, vocab)
            np.save(out + ".ids.npy", ids)
            np.save(out + ".offs.npy", offs)
            print(f"  wrote {out}: {len(ids):,} keys, {len(offs)-1:,} records")
    with open(f"{out_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab.to_id, f)
    print(f"vocab size: {len(vocab):,}")


def load_vocab(out_dir: str) -> KeyVocab:
    v = KeyVocab()
    with open(f"{out_dir}/vocab.json", encoding="utf-8") as f:
        v.to_id = json.load(f)
    return v


def load_keys(out_dir: str, split: str, s: int):
    ids = np.load(f"{out_dir}/{split}_source{s}.ids.npy")
    offs = np.load(f"{out_dir}/{split}_source{s}.offs.npy")
    return ids, offs


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else "work/cache"
    out = sys.argv[2] if len(sys.argv) > 2 else "work/keys"
    build_cache(cache, out)
