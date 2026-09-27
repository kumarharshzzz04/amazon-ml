"""
Stage 5 - inference: score test candidates and write the submission TSVs.

Outputs (atomic via .tmp + os.replace):
  output/matching_results.tsv   source1_entity_id <TAB> comma-joined matches
  output/candidate_pairs.tsv    source1_entity_id <TAB> comma-joined candidates

Every test S1 row gets exactly one line (empty list for predicted singletons),
and emitted matches are always a subset of the emitted candidates.  An S1 row
with zero candidates is written with an empty list (true singleton policy).

Usage:
  python -m src.infer <blob_dir> <cand_dir> <model_dir> <out_dir> [thr]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feats_fast import BlobStore, compute_chunk, NF  # noqa: E402

CH = 150_000        # pairs per worker job
BLOCK = 20          # windows (CH pairs) per predict+write block


def _winit(blob_dir: str, split: str) -> None:
    global _S
    _S = BlobStore(blob_dir, split)


def _wjob(args):
    blob_dir, split, i1s, ts, ksc, kcn = args
    store = _S if "_S" in globals() else BlobStore(blob_dir, split)
    return compute_chunk(store, list(zip(i1s, ts, ksc, kcn)))


def _best_iter(model_dir: str):
    meta_p = f"{model_dir}/meta.json"
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as f:
            bi = json.load(f).get("best_iteration") or 0
        return bi or None
    return None


def _load_threshold(model_dir: str, thr_arg: float | None) -> float:
    if thr_arg is not None:
        return float(thr_arg)
    p = f"{model_dir}/threshold.json"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    print("WARNING: no threshold.json; using 0.5", file=sys.stderr)
    return 0.5


def run(blob_dir: str, cand_dir: str, model_dir: str, out_dir: str,
        thr_arg: float | None = None, workers: int = 6,
        cache_dir: str = "work/cache") -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    import lightgbm as lgb
    import pyarrow.parquet as pq

    flat = np.load(f"{cand_dir}/cand.flat.npy", mmap_mode="r")
    offs = np.load(f"{cand_dir}/cand.offs.npy", mmap_mode="r")
    score = np.load(f"{cand_dir}/cand.score.npy", mmap_mode="r")
    cnt = np.load(f"{cand_dir}/cand.count.npy", mmap_mode="r")
    n_s1 = len(offs) - 1
    thr = _load_threshold(model_dir, thr_arg)
    print(f"* {n_s1:,} test S1, thr={thr} ({time.time()-t0:.0f}s)",
          file=sys.stderr)

    # target id strings in concat row order (S2 rows then S3 rows)
    tids: list[str] = []
    for s in (2, 3):
        for b in pq.ParquetFile(
                f"{cache_dir}/test_source{s}.parquet").iter_batches(
                batch_size=500_000, columns=["entity_id"]):
            tids.extend(b.column("entity_id").to_pylist())
    s1_ids: list[str] = []
    for b in pq.ParquetFile(
            f"{cache_dir}/test_source1.parquet").iter_batches(
            batch_size=500_000, columns=["entity_id"]):
        s1_ids.extend(b.column("entity_id").to_pylist())
    assert len(s1_ids) == n_s1, (len(s1_ids), n_s1)
    print(f"* ids loaded ({time.time()-t0:.0f}s)", file=sys.stderr)

    booster = lgb.Booster(model_file=f"{model_dir}/model.txt")
    bi = _best_iter(model_dir)

    mt_tmp = f"{out_dir}/matching_results.tsv.tmp"
    cd_tmp = f"{out_dir}/candidate_pairs.tsv.tmp"
    mf = open(mt_tmp, "w", encoding="utf-8", newline="")
    cf = open(cd_tmp, "w", encoding="utf-8", newline="")
    mf.write("source1_entity_id\tmatched_entity_ids\n")
    cf.write("source1_entity_id\tcandidate_entity_ids\n")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers, initializer=_winit, initargs=(blob_dir, "test"))

    n_match, n_nonempty, n_pairs_done = 0, 0, 0
    try:
        for blo in range(0, n_s1, 200_000):
            bhi = min(blo + 200_000, n_s1)
            pa_start = int(offs[blo])
            pa_end = int(offs[bhi])
            n_blk = pa_end - pa_start
            if n_blk == 0:
                for i in range(blo, bhi):
                    mf.write(f"{s1_ids[i]}\t\n")
                    cf.write(f"{s1_ids[i]}\t\n")
                continue

            blk_feats = np.empty((n_blk, NF), dtype=np.float32)
            blk_s1_full = np.repeat(np.arange(blo, bhi, dtype=np.int64),
                                    np.diff(np.asarray(offs[blo:bhi + 1])))

            def _jobs():
                for a in range(pa_start, pa_end, CH):
                    b = min(a + CH, pa_end)
                    yield (blob_dir, "test",
                           blk_s1_full[a - pa_start:b - pa_start].tolist(),
                           flat[a:b].tolist(), np.asarray(score[a:b]).tolist(),
                           np.asarray(cnt[a:b]).tolist())

            n_wins = (n_blk + CH - 1) // CH
            for wi, res in enumerate(pool.imap(_wjob, _jobs(), chunksize=1)):
                a = pa_start + wi * CH
                b = min(a + CH, pa_end)
                blk_feats[a - pa_start:b - pa_start] = res
                n_pairs_done += b - a

            preds = booster.predict(blk_feats, num_iteration=bi)
            del blk_feats

            # segment ids per pair within block
            blk_s1 = blk_s1_full
            blk_t = np.asarray(flat[pa_start:pa_end])
            keep = preds >= thr
            order = np.argsort(blk_s1, kind="stable")
            s1k, tk, kk = blk_s1[order], blk_t[order], keep[order]
            uniq, starts = np.unique(s1k, return_index=True)
            starts = list(starts) + [len(s1k)]
            gmap = {int(u): (starts[i], starts[i + 1])
                    for i, u in enumerate(uniq)}

            for s in range(blo, bhi):
                g = gmap.get(s)
                if g is None:            # zero candidates for this row
                    mf.write(f"{s1_ids[s]}\t\n")
                    cf.write(f"{s1_ids[s]}\t\n")
                    continue
                ga, gb = g
                cand_ids = [tids[int(x)] for x in tk[ga:gb]]
                cf.write(s1_ids[s] + "\t" + ",".join(cand_ids) + "\n")
                sel = [cand_ids[i] for i in range(gb - ga) if kk[ga + i]]
                if sel:
                    mf.write(s1_ids[s] + "\t" + ",".join(sel) + "\n")
                    n_nonempty += 1
                    n_match += len(sel)
                else:
                    mf.write(f"{s1_ids[s]}\t\n")
            if (blo // 200_000) % 5 == 0:
                print(f"  s1 {bhi:,}/{n_s1:,} pairs={n_pairs_done:,} "
                      f"matches={n_match:,} ({time.time()-t0:.0f}s)",
                      file=sys.stderr)
    finally:
        pool.terminate()
        mf.close()
        cf.close()
    os.replace(mt_tmp, f"{out_dir}/matching_results.tsv")
    os.replace(cd_tmp, f"{out_dir}/candidate_pairs.tsv")
    print(f"* wrote outputs: {n_nonempty:,} S1 with >=1 match, "
          f"{n_match:,} total matches ({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2], a[3], a[4],
        thr_arg=float(a[5]) if len(a) > 5 else None)
