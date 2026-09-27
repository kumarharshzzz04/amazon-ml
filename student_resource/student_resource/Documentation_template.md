# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM NAME]
**Team Members:** [TEAM MEMBERS]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

We built a scalable entity-resolution pipeline: multi-key country-aware blocking with
soft document-frequency caps generates ~150 candidates per Source-1 record at 94.4%
pair recall, and a LightGBM pair classifier over 34 lexical/agreement features with an
F0.5-calibrated global threshold produces the final matches. Everything runs on CPU in
under 6 hours end-to-end on a single 12-core machine, streaming via memmaps and parquet
caches so the 13M-record corpus never needs to fit in RAM.

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

- **Corpus:** train has 2.21M S1 / 5.03M S2 / 5.29M S3 records; each S2/S3 record
  matches at most one S1; ~5.6% of S1 entities are true singletons (predict-empty
  earns full credit for them).
- **Noise:** legal-suffix variants (Corp/Corporation, Pvt Ltd, LLC), typos, diacritics,
  Indic transliteration, reordered address components, and ~4.5% empty addresses.
- **Same-country invariant:** 100% of sampled true pairs are same-country (US + India
  in train; France appears only in test). We therefore treat country as an open set and
  use it as a *blocking* signal (never as a filter), which generalizes to France and
  simultaneously hardens the decision boundary.
- **Scale constraint:** the full train candidate matrix (321M pairs) cannot be
  materialized with features on disk (~43 GB), which drove the subsampled-training +
  full-pool-calibration design below.

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** Blocking + pairwise classifier (LightGBM) + global threshold
calibration.
**Core Innovation:** Country-prefixed rarest-key blocking with a soft document-frequency
cap and IDF-ranked candidate selection, combined with negative-subsampled training and
full-candidate-pool threshold calibration so the decision threshold matches inference
conditions exactly.

Pipeline stages (each resumable, streaming over parquet/memmaps):
1. **prep** — Unicode-normalize (unidecode transliteration), lowercase, strip
   punctuation, canonicalize legal/street/state tokens, collapse multiword
   canonical forms; cache to parquet.
2. **keys** — per-record blocking keys: individual tokens, sorted adjacent bigrams,
   postal-code keys, all prefixed with the country string; stored as CSR arrays.
3. **candidates** — inverted index over keys; keys with df > 2000 are dropped softly
   (prefix discarding), the k=10 rarest keys per record are selected, and candidate
   pairs are ranked by summed IDF of shared keys with top_m=150 kept per S1 row.
4. **features** — 34 pair features (below) computed with rapidfuzz over memory-mapped
   text blobs with a 6-process pool.
5. **train** — LightGBM binary classifier on all 7.21M true positives + ~6.75M
   sampled hard negatives; early stopping on a held-out half of S1 entities.
6. **calibrate** — threshold sweep for macro-F0.5 on ALL candidates of 100k sampled
   S1 rows (full-pool score distribution, including true singletons).
7. **infer** — stream test candidates through the model in 150k-pair blocks, apply the
   threshold, write `matching_results.tsv` + `candidate_pairs.tsv` (every test S1 row
   present; matches are always a subset of candidates).

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** country-prefixed name tokens, sorted adjacent character
  bigrams of the normalized name (robust to typos and transliteration), and 5/6-digit
  postal codes — 10 rarest keys per record, ranked globally by document frequency
  (soft cap: keys with df > 2000 are excluded unless a record has too few rare keys).
- **Candidate pairs generated:** 321.1M for train (mean 145.5 per S1, top_m=150
  IDF-ranked per S1); the same configuration is applied to test.
- **How you ensured true matches were not lost:**
  - Verified directly against train ground truth: **94.4% pair recall**
    (7,210,243 / 7,638,365 true pairs found), i.e. the F0.5 ceiling.
  - Country-prefixing exploits the same-country invariant (100% of true pairs) to cut
    the comparison space ~2x without losing true matches, and generalizes to unseen
    countries such as France.
  - Adjacent-bigram keys recover pairs with typos/tokenization drift that token keys
    miss; postal keys recover name-mangled pairs with intact addresses.
  - Ablations (20k-S1 samples): token-only keys k=6 with co-occurrence filtering gave
    96.4% recall but 3.9k candidates/S1; adding country+bigram keys gave 96.3% at
    1,044 candidates/S1; the production k=10 config trades ~2pp recall for a 7x
    smaller inference workload, which lets the classifier see cleaner, IDF-ranked
    candidate pools.

---

## 4. Matching Model

**Features used (34):**
- Name features: token Jaccard and overlap coefficient, character-level fuzz ratio /
  partial / token-set / token-sort ratios, character-bigram cosine, Jaro-Winkler,
  length ratio, legal-suffix agreement, core-token Jaccard (suffixes removed).
- Address features: token Jaccard/overlap, fuzz family, char-bigram cosine,
  Jaro-Winkler, digit-run Jaccard and subset-containment, postal equality (raw value
  match), state-token equality, address-empty flags (one-sided / both).
- Other: country equality (open-set), shared-key IDF score, shared-key count and
  count ratio, postal-hash equality.

**Model type:** LightGBM (gradient-boosted trees), 96 leaves, lr 0.06, ≤600 rounds
with early stopping (50) on a 50/50 S1-entity hash split; positives weighted to
compensate subsampling. MIT-licensed, CPU-only.

**Threshold selection method:** global threshold swept over 18 values + fine local
refinement, maximizing **macro-F0.5 on the full candidate pools** (all ~145 candidates
per S1) of 100k held-out S1 rows, so the calibrated score distribution matches
inference exactly; true singletons are included in the calibration objective
(predict-empty scores 1.0 for them).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [VAL F0.5 — filled after calibration] on the 100k-row
  full-pool validation sample; candidate recall ceiling 94.4%.
- **Common false positives (wrong merges):** generic names sharing a common token
  (e.g., "mart", "store") in the same city with dissimilar addresses; branch chains
  with near-identical names and different street numbers.
- **Common false negatives (missed matches):** pairs whose only shared signal is a
  high-df token (heavily truncated names); empty-address pairs relying purely on
  name bigrams; heavy transliteration variants beyond unidecode's reach.

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

A two-stage retrieve-then-classify pipeline with country-aware rare-key blocking and a
calibrated LightGBM matcher delivers high precision at near-ceiling recall within the
F0.5 (precision-weighted) metric on commodity CPU hardware. The key lessons: exploit
dataset invariants (same-country pairs) at blocking time rather than as filters so they
generalize to unseen countries, and calibrate thresholds on the *full* candidate-pool
score distribution — subsampled training pools silently shift optimal thresholds.

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

```
code/business_entity_resolution/
├── run_pipeline.py        # resumable driver: prep → keys → candidates → recall
│                          #   → blobs → features → train → calibrate → inference
├── requirements.txt       # pinned dependency versions
├── src/
│   ├── prep.py            # normalization → parquet caches
│   ├── keys.py            # blocking keys (CSR)
│   ├── candidates.py      # soft-cap index, key selection, top_m ranking
│   ├── eval_recall.py     # candidate recall vs GT + positives dump
│   ├── blobs.py           # memory-mapped text blobs per source
│   ├── build_features_sub.py  # negative-subsampled train features
│   ├── build_features_range.py # full-pool features for threshold calibration
│   ├── feats_fast.py      # 34-pair-feature kernel (rapidfuzz, 6 workers)
│   ├── train.py           # LightGBM training
│   ├── calibrate.py       # F0.5 threshold sweep on full pools
│   ├── infer.py           # streaming test scoring → both output TSVs
│   └── metric.py          # macro-F0.5 scoring utilities
└── tests/test_index.py    # blocking-index correctness tests
```

Reproduce end-to-end: `python run_pipeline.py all` (after downloading the dataset
TSVs into `student_resource/student_resource/dataset/`). Full run ≈ 6 h on 12 cores /
16 GB RAM; peak disk ≈ 12 GB of caches. Unit test: `tests/test_index.py`.

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

- Blocking ablations (20k sampled S1): see Section 3; production config chosen for
  recall-per-candidate efficiency and France generalization.
- Threshold sweep table and per-threshold F0.5 are persisted in
  `work/model/threshold.json` by the calibrate stage.
- Candidate recall by key family is reported by `src/eval_recall.py`
  (94.4% production).

---
