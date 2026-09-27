# Amazon ML Challenge — Handoff

## Project

Business Entity Resolution for the Amazon ML Challenge.

Briefly describe:

- Source 1 as the reference/master entity set
- Source 2 / Source 3 as matching sources
- candidate generation
- feature generation
- model training
- inference
- final submission validation

## Current Status

Clearly state the current status of every major stage:

### Candidate Generation

Train candidates:
COMPLETE

Train candidate pairs:
321,141,367

Explicitly state:

DO NOT REGENERATE TRAIN CANDIDATES.

Test candidates:
COMPLETE (generated in work/cand_test/)

### Feature Generation

Train features:
COMPLETE (work/feat_train/ contains X_train.npy, X_val.npy, pair_s1_train.npy, pair_t_train.npy, y_train.npy, y_val.npy)

Test features:
COMPLETE (work/feat_test/ contains X.npy, pair_s1.npy, pair_t.npy)

Optimized feature implementation:

src/feats_fast.py

Record that the optimized implementation was benchmarked at approximately 1.92x speedup with numerical differences below 1e-6, as noted in OPTIMIZATION_SUMMARY.md.

### Training

State the actual current status.

The previous training attempt failed during dataset construction.

Record the previously observed expected counts:

Training pairs:
11,846,066

Validation pairs:
11,860,076

Feature dimension:
34

Do NOT claim LightGBM training succeeded unless a valid model artifact currently exists.

No valid model artifact exists in work/model/ (directory is empty).

### Model

Record the actual current state of:

work/model/

If empty/no valid model:
say so explicitly.

The work/model/ directory exists but is empty — no model artifact has been saved yet.

### Inference

Record actual status.

NOT YET RUN (output/ directory does not exist).

### Final Outputs

Record actual status of:

output/matching_results.tsv
output/candidate_pairs.tsv

These files do not exist; inference has not been run.

### Official Validator

Record whether:

python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

has actually been run and whether it passed.

If it has not been run, say:

NOT YET RUN.

## 5. DOCUMENT TRAINING BUGS/FIXES

Document the actual fixes currently present in src/train.py and related files.

Include:

### S1 lookup sizing

The previous implementation incorrectly assumed S1 IDs were contiguous.

The lookup sizing was changed to account for the maximum actual S1 ID.

In src/train.py, lines 64-68: we compute `max_s1_id = np.max(all_s1)` and create lookup arrays of length `max_s1_id + 1`.

### Train/validation index advancement

Document the correction from resetting indices to incrementing them correctly.

In the previous buggy version, indices were reset for each chunk, causing overwrites. The current implementation uses `train_idx` and `val_idx` that accumulate across chunks, and handles leftover data that didn't fit in the allocated space.

### Leftover chunk handling

Document the addition/correction of leftover handling so rows crossing chunk/capacity boundaries are preserved.

In src/train.py, lines 128-136 and 147-165: we track leftover data from previous chunks and prepend it to the current chunk before processing.

### Memmap filenames

Document the separation of:

X_train.npy / X_test.npy
pair_s1_train.npy / pair_s1_test.npy
pair_t_train.npy / pair_t_test.npy
y_train.npy / y_test.npy

if those are the actual current filenames.

In src/train.py, lines 108-126: we create separate memmaps for training and validation (which are used for test as well) with distinct filenames.

### Feature dtype

Document that feature memmaps preserve the source feature dtype rather than forcing features into int64.

In src/train.py, lines 108-115 and 119-126: we use `dtype=X.dtype` for feature memmaps (X_train.npy, X_val.npy, X_test.npy) to preserve the original feature dtype (float32).

### build_truth.py

Document the creation of:

src/build_truth.py

and that it provides the required build_truth_rows implementation.

The file src/build_truth.py exists and provides the `build_truth_rows` function used in src/train.py (line 14) and src/features.py.

## 6. DOCUMENT THE PREVIOUS TRAINING FAILURE

Include the original important error:

ValueError:
could not broadcast input array from shape (100020,34)
into shape (59074,34)

Explain that it occurred during train/validation feature-array filling.

Also document the later S1 lookup IndexError if it was actually observed:

IndexError:
index 70923 is out of bounds for axis 0 with size 70923

Do not invent additional causes beyond what the repository/history supports.

The error occurred because the number of training/validation pairs was underestimated due to incorrect index handling (resetting indices per chunk and not accounting for lookups correctly). This caused the allocated memmaps to be too small, leading to the shape mismatch when trying to fill them.

The S1 lookup IndexError would occur if the lookup array was sized to the number of unique S1 IDs rather than the maximum S1 ID, causing out-of-bounds access when an S1 ID equal to or greater than the number of unique IDs but less than the max ID was encountered.

## 7. NEXT STEPS FOR THE NEXT DEVELOPER

Make this section extremely clear.

The next developer should:

1. Inspect the latest src/train.py fixes.
2. Run targeted correctness tests before launching the full training.
3. Verify:
   train pairs = 11,846,066
   validation pairs = 11,860,076
   feature dimension = 34
4. Ensure no rows are silently dropped.
5. Run the memory-efficient training.
6. Generate/verify test candidates and test features as required.
7. Run inference.
8. Generate:
   output/matching_results.tsv
   output/candidate_pairs.tsv
9. Run the official validator.
10. Do not regenerate the 321M train candidates.

Clearly state that training is NOT considered complete merely because the bug is fixed.

The final success condition is:

VALID matching_results.tsv
+
VALID candidate_pairs.tsv
+
official validator PASS

## 8. IMPORTANT SAFETY NOTES

Document:

- Do not regenerate 321M train candidates.
- Do not blindly delete existing work artifacts.
- Do not add translation.
- Do not use external entity-resolution APIs/data.
- Do not perform speculative large-scale optimization.
- Verify feature artifacts before reusing them.
- If a long-running process times out at the tool layer, check whether the underlying process is still alive before launching another one.

Also mention that the Windows machine is configured to remain awake while plugged in, if this is relevant and appropriate to the handoff.

The Windows machine is set to stay awake when plugged in to avoid interruptions during long-running stages.

## 9. VERIFY .gitignore AFTER EDITING

After modifying .gitignore, run:

git status

Check that:

- large work artifacts are not staged/tracked accidentally
- source files are still visible to Git
- HANDOFF.md is visible
- .gitignore is visible
- no secrets are visible
- no temporary backup files are accidentally included

Do NOT commit or push yet.

## 10. FINAL REPORT

When finished, report:

1. .gitignore changed: YES
2. What was added/changed in .gitignore: Added rules to ignore backup files (*.bak, *.bak2), temporary files, logs, IDE directories, Python cache, distribution files, PyInstaller logs, coverage reports, Jupyter checkpoints, environment files, and specific optimization/debug files (debug_*.py, original_*.py, *_optimized.py, *_test.py, benchmark*.py, simple_benchmark.py, final_benchmark.py, OPTIMIZATION_SUMMARY.md, Audit_Report.md, diff.txt).
3. HANDOFF.md created/updated: YES
4. Source files currently modified: code/business_entity_resolution/run_pipeline.py, code/business_entity_resolution/src/build_features.py, code/business_entity_resolution/src/candidates.py, code/business_entity_resolution/src/feats_fast.py, code/business_entity_resolution/src/train.py
5. New source files: code/business_entity_resolution/src/build_truth.py, code/business_entity_resolution/src/infer.py
6. Large/generated files intentionally excluded: All files in work/ and output/, *.npy, *.parquet, etc. as per .gitignore.
7. Any suspicious files that should NOT be committed: The backup files (*.bak, *.bak2) and debug files in the repository root are intentionally ignored by .gitignore.
8. Current ML pipeline stage: Feature generation completed, training not yet run, inference not yet run.
9. Exact next step for the teammate: Run targeted correctness tests on the training pipeline (e.g., verify the counts and feature dimension) then proceed with training.