# Team Handoff — Amazon ML Challenge 2026 (Business Entity Resolution)

## 🫵 WHERE TO START (new teammate: read this block, then nothing else until running)

You are taking over a half-finished, fully-documented pipeline. Do these in order:

1. **Env check (2 min):** Python 3.11+ with
   `pip install -r code/business_entity_resolution/requirements.txt`.
   Windows: use the real python (`C:\Python3xx\python.exe`), never the
   `python3` store-stub, and always set `PYTHONIOENCODING=utf-8`.
2. **Data (10 min):** download the challenge dataset zip from the portal,
   unzip so `student_resource/student_resource/dataset/{train,test}/*.tsv` exist.
3. **Verify (1 min):**
   `python code/business_entity_resolution/tests/test_index.py`
   → must end with `ALL INDEX TESTS PASSED`.  (This proves the index fix is
   active — see Safety Rules below.)
4. **Run (hours, unattended):**
   `python code/business_entity_resolution/run_pipeline.py all`
   Every finished stage auto-skips. On a fresh clone with data downloaded, the
   first run rebuilds stages 1–3 (~1.5–2 h) then continues through training and
   inference on its own. Walk away; monitor with `tail -f work/<latest>.log`.
5. **Validate & submit:**
   `python student_resource/student_resource/utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir student_resource/student_resource/dataset/test`
   It must print `PASS`. Then upload `output/matching_results.tsv` to the portal.

### If the previous machine's big files are available (Google Drive etc.)
You can skip stages 1–3 by dropping their folders into `work/`:
`work/cache/` (918 MB), `work/keys2/` (3.0 GB), `work/cand_train/` (~3.3 GB).
Then `run_pipeline.py all` jumps straight to the remaining stages.

### Division of labor (if more than one of you)
* Machine A: `run_pipeline.py all` (train track → submission files)
* Machine B (independent, parallel): `run_pipeline.py candidates test`
  — saves ~1 h; then whoever reaches inference first uses it.
* Keep this file updated (status table + anything you learn). That is the
  whole handoff protocol.

---

**Goal:** for each Source-1 record, find every matching Source-2/Source-3 record.
**Metric:** macro-F0.5 per S1 entity (precision-weighted 2:1, singletons included).
**Deadline:** Sept 27, 11:59 PM IST — 5 submissions/day per team.

---

## Current status (updated Sept 26)

| Stage | State | Output |
|---|---|---|
| 1. prep (normalize 24M records) | ✅ done | `work/cache/*.parquet` |
| 2. keys (blocking keys, 27M vocab) | ✅ done | `work/keys2/` |
| 3. candidates train (k=10, top_m=150) | 🔄 running now (~50%) | `work/cand_train/cand.*.npy` |
| 4. recall eval + positive pairs | ⬜ run after stage 3 (STALE FILES DELETED - old ones were from the buggy index, do not reuse) | `work/cand_train/pos_*.npy` |
| 5. blobs (text memory-maps) | ⬜ script ready | `work/blobs/` |
| 6. features | ⬜ script ready | `work/feat_train/` |
| 7. LightGBM + threshold | ⬜ script ready | `work/model/` |
| 8. test candidates + inference | ⬜ script ready | `output/*.tsv` |
| 9. validate + zip | ⬜ | submission package |

### ⚠️ HANDOFF SAFETY RULES (read first!)

1. **Never trust artifacts from before the index fix.** If you find
   `work/cand_train/pos_s1.npy` or `pos_t.npy` already present, DELETE them and
   re-run stage `recall` — they must be regenerated from the FIXED candidates.
   (`run_pipeline.py recall` does this automatically once the old files are gone.)
2. `run_pipeline.py` skips any stage whose output file exists — so before
   re-running anything after this handoff, verify the outputs you have are from
   the fixed code (candidates written after 14:30 local Sept 26 are safe).
3. The index correctness test lives in the repo:
   `code/business_entity_resolution/tests/test_index.py` — run it (must print
   `ALL INDEX TESTS PASSED`) before trusting any candidate output.

---

## Quick start (any teammate, any machine)

```bash
# 1. clone this repo (only ~50KB of code - all data is gitignored)
git clone <repo-url> && cd amazon-ml

# 2. python 3.11+ with the pinned deps
python -m pip install -r requirements.txt

# 3. download the dataset zip from the challenge portal and unzip it so that
#    student_resource/student_resource/dataset/{train,test}/*.tsv exists

# 4. run whichever stage is unfinished (each stage skips itself if done):
python code/business_entity_resolution/run_pipeline.py all
```

Stages are independent — with several machines you can run in parallel:

```bash
# machine A (after candidates-train is committed to a shared drive or rerun):
python code/business_entity_resolution/run_pipeline.py blobs
python code/business_entity_resolution/run_pipeline.py features train
python code/business_entity_resolution/run_pipeline.py train

# machine B (fully independent of machine A):
python code/business_entity_resolution/run_pipeline.py candidates test
```

**Disk warning:** stage 3+6 want ~25GB free per split. The default NTFS/laptop
with 12GB free must run stages one at a time with cleanup between (that is what
`work/cleanup.py` below automates).

---

## Critical environment notes (read before debugging!)

* **Windows:** use the real `python` (e.g. `C:\Python314\python.exe`). The
  `python3` alias is a broken Microsoft-store stub. Always set
  `PYTHONIOENCODING=utf-8` — printing Devanagari to a cp1252 console will crash.
* **Long jobs:** run them backgrounded (`nohup python ... &`) and monitor via
  `tail -f work/<log>` — a sync shell call dies at 10 minutes.
* **Git:** `work/` and `output/` are gitignored on purpose (gigabytes of `.npy`).
  Teammates share *code* through git and *artifacts* by re-running stages or a
  shared drive (Google Drive / LAN share) for the big intermediate `.npy` files.

---

## Technical summary (what the pipeline does)

1. **prep** — normalize names/addresses: lower-case, strip accents, unidecode
   transliteration (Indic scripts → Latin), canonicalize legal suffixes
   (Corp/Corporation→corp), street types (Road/Rd→rd), US/India/France states,
   multiword states ("new york"→ny).  Output: 6 parquet caches, 24.2M records.
2. **keys** — per record emit blocking keys: country-prefixed name/addr tokens,
   sorted adjacent token bigrams (word-order robust), digit-signature, postal
   code.  Key vocab ~27M; stored as CSR int32 arrays.
3. **candidates** — build an inverted index (CSR) over S2+S3 targets; per S1
   record take the k=10 rarest keys (soft df cap 2000), union their postings,
   IDF-score pairs, keep top 150 per S1.  Empirical recall 93.3% at k=6 —
   k=10 target ≥95%.
   **BUG FIXED this session:** s3 rows were unoffset in postings (collided into
   s2 row-space) and the two sources were sorted separately breaking global key
   order — both now verified with a 4-record synthetic test in
   `work/diag_*.py`.
4. **features** — 34 dims per (S1,target) pair: token jaccard/overlap, rapidfuzz
   ratio/partial/token_set/token_sort, char-bigram cosine, Jaro-Winkler, legal
   suffix agreement, digit/POSTAL/state agreement, idf score, shared-key count.
   Computed from memory-mapped text blobs (never loads 24M strings into RAM).
5. **train** — 50/50 hash split of S1 rows; LightGBM binary classifier
   (positives = ground-truth pairs among candidates); F0.5-threshold sweep on
   the validation half; saves `model.txt` + `threshold.json`.
6. **inference** — candidates for test S1 (1.73M), stream features, model
   scores, threshold, write `output/matching_results.tsv` (scored) +
   `output/candidate_pairs.tsv` (audit), then run the official validator.

---

## Rules compliance (verified)

* ❌ No external data/APIs/geocoding — grep-clean, stdlib+pip only.
* ❌ No country hardcoding — country is an open-set string; France flows
  through the same path (state table leaves unknown tokens untouched).
* ✅ Model: LightGBM (MIT), tiny parameter count (≪ 8B cap).
* ✅ Full dataset used (24.2M records verified against raw file row counts).
* ✅ Output format enforced by `utils/validate_submission.py` before upload.

## Ideas if there is time (ranked by expected F0.5 gain)

1. **Raise candidate recall to ≥97%** — the single biggest lever. Options:
   k=12 with hard_cap, add 'near-duplicate zip key', per-country df caps.
2. **Isotonic/precision calibration per candidate-count bucket** — entities
   with 150 candidates need a stricter threshold than ones with 5.
3. **Manual rules on top of the model** for the singleton guard: if the best
   pair's name-token jaccard < 0.2 and postal mismatch → force empty.
4. GPU fine-tuned multilingual matcher (transformer) for the Indic/French name
   noise — big jump potential but only for round 2 (RTX 4050 available).

## Files map

```
code/business_entity_resolution/
  run_pipeline.py       ← one-command driver (resume-safe)
  src/normalize.py      ← all text normalization + lexical tables
  src/prep.py           ← stage 1
  src/keys.py           ← stage 2 (word+bigram+postal keys)
  src/blocking.py       ← select_keys / candidates_for primitives
  src/candidates.py     ← stage 3 (inverted index + IDF ranking)
  src/eval_recall.py    ← stage 4 (recall + positive pairs)
  src/blobs.py          ← stage 5 (memory-mapped text)
  src/build_features.py ← stage 6 driver
  src/feats_fast.py     ← stage 6 compute (34 features)
  src/train.py          ← stage 7 (LightGBM + F0.5 calibration)
  src/infer.py          ← stage 8 (test inference → output TSVs)
  src/metric.py         ← F0.5 per entity + threshold helpers
work/STATUS.md          ← progress log maintained through the session
utils/validate_submission.py   (provided by organizers, in student_resource)
```
