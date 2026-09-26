"""
Extended blocking-key extraction: word keys + character n-gram keys.

Char n-grams (3-5) over the normalized name+addr are added as an additional key
class ``c``.  They are robust to typos, transpositions, spacing variants and
merged/split words, which word tokens are not.  Both classes live in the same
df-ranked key space, so rarest-key selection automatically prefers whichever
anchor is more distinctive for a record.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import keys as K  # noqa: E402
from src import normalize as N  # noqa: E402


def char_ngrams(text: str, sizes=(3, 4, 5), cap: int = 40) -> set[str]:
    """Character n-grams of the normalized text, with word-boundary padding."""
    t = " " + N.norm_text(text) + " "
    grams: set[str] = set()
    for sz in sizes:
        if len(t) < sz:
            if t.strip():
                grams.add(t.strip())
            continue
        for i in range(len(t) - sz + 1):
            grams.add(t[i:i + sz])
        if len(grams) > cap * 3:
            break
    return grams


def record_keys2(name: str, addr: str) -> set[str]:
    """Word keys (class n/a/D/z) plus char n-gram keys (class c)."""
    out = K.record_keys(name, addr)
    grams = char_ngrams(name, cap=24) | char_ngrams(addr, cap=24)
    for g in grams:
        out.add("c" + g)
    return out


class KeyVocab2:
    """String key -> int id, built incrementally (extended key space)."""

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


def encode_source2(parquet_path: str, vocab: KeyVocab2) -> tuple[np.ndarray, np.ndarray]:
    """Encode one source into (flat_ids, offsets) with extended keys."""
    import pyarrow.parquet as pq

    ids: list[int] = []
    offsets: list[int] = [0]
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    seen = 0
    t0 = time.time()
    for b in pf.iter_batches(batch_size=200_000, columns=["name_norm", "addr_norm"]):
        for n, a in zip(b.column("name_norm").to_pylist(),
                        b.column("addr_norm").to_pylist()):
            for k in record_keys2(n, a):
                ids.append(vocab.intern(k))
            offsets.append(len(ids))
        seen += len(offsets) - 1
        print(f"    {os.path.basename(parquet_path)} {seen:,}/{total:,} "
              f"({time.time()-t0:.0f}s)", file=sys.stderr)
    return np.asarray(ids, dtype=np.int32), np.asarray(offsets, dtype=np.int64)


def build_cache2(cache_dir: str, out_dir: str, splits=("train", "test")) -> None:
    os.makedirs(out_dir, exist_ok=True)
    vocab = KeyVocab2()
    for split in splits:
        for s in (1, 2, 3):
            src = f"{cache_dir}/{split}_source{s}.parquet"
            print(f"* encoding {src}")
            ids, offs = encode_source2(src, vocab)
            np.save(f"{out_dir}/{split}_source{s}.ids.npy", ids)
            np.save(f"{out_dir}/{split}_source{s}.offs.npy", offs)
            print(f"  wrote {split}_source{s}: {len(ids):,} keys, "
                  f"{len(offs)-1:,} records")
    with open(f"{out_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab.to_id, f)
    print(f"vocab size: {len(vocab):,}")


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else "work/cache"
    out = sys.argv[2] if len(sys.argv) > 2 else "work/keys2"
    build_cache2(cache, out)
