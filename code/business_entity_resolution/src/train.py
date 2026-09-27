"""
Training script for business entity resolution model.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import normalize as N  # noqa: E402
from src.features import NF  # noqa: E402
from src.build_truth import build_truth_rows


def main(feature_dir: str, truth_dir: str, ground_truth_path: str, cache_dir: str,
         model_dir: str) -> None:
    """Main training function."""
    # Load precomputed features and metadata
    print(f"Loading features from {feature_dir}...", file=sys.stderr)
    pair_s1 = np.load(f"{feature_dir}/pair_s1.npy", mmap_mode="r")
    pair_t = np.load(f"{feature_dir}/pair_t.npy", mmap_mode="r")
    X = np.load(f"{feature_dir}/X.npy", mmap_mode="r")
    n_pairs = X.shape[0]
    print(f"  Loaded {n_pairs:,} pairs, {X.shape[1]} features", file=sys.stderr)

    # Build truth dictionary: maps S1 row indices to sets of target indices
    print(f"Building truth dictionary from {ground_truth_path}...", file=sys.stderr)
    truth, _ = build_truth_rows(truth_dir, ground_truth_path)
    print(f"  Loaded truth for {len(truth):,} S1 rows", file=sys.stderr)

    # First pass: find all unique S1 IDs and their frequency
    print("  Finding unique S1 rows...", file=sys.stderr)
    chunk_size = 1000000  # 1M pairs per chunk
    unique_s1_set = set()
    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        s1_chunk = pair_s1[start:end]
        unique_s1_set.update(np.unique(s1_chunk))
        if start % (10 * chunk_size) == 0:
            print(f"    processed {start:,} pairs, found {len(unique_s1_set):,} unique S1 so far",
                  file=sys.stderr)

    all_s1 = np.array(list(unique_s1_set))
    print(f"  Total unique S1 rows: {len(all_s1):,}", file=sys.stderr)

    # deterministic 10/10 split of S1 rows (20% total, 10% train, 10% val)
    rng = np.random.default_rng(42)
    indices = np.arange(len(all_s1))
    rng.shuffle(indices)
    n_train_s1 = int(0.1 * len(all_s1))
    n_val_s1 = int(0.1 * len(all_s1))
    # Get indices for train/val split
    train_indices = indices[:n_train_s1]
    val_indices = indices[n_train_s1:n_train_s1+n_val_s1]
    # Get actual S1 IDs for train/val using advanced indexing
    train_s1_arr = all_s1[train_indices]
    val_s1_arr = all_s1[val_indices]
    print(f"  S1 rows: {len(all_s1):,} (train: {len(train_s1_arr):,}, val: {len(val_s1_arr):,})",
          file=sys.stderr)

    # Create lookup arrays for fast membership testing
    max_s1_id = np.max(all_s1)
    train_s1_lookup = np.zeros(max_s1_id + 1, dtype=bool)
    train_s1_lookup[train_s1_arr] = True
    val_s1_lookup = np.zeros(max_s1_id + 1, dtype=bool)
    val_s1_lookup[val_s1_arr] = True

    # First pass: count training and validation pairs and build truth lookup
    n_train = 0
    n_val = 0
    # Build keys_truth directly from truth to avoid intermediate structures
    keys_list = []
    for i1, rows in truth.items():
        for target_id in rows:
            keys_list.append((target_id << np.int64(32)) | np.int64(i1))
    keys_truth = np.array(keys_list, dtype=np.int64)
    # keys_truth.sort()  # Removed as keys_truth has no duplicates and we'll use assume_unique=True in np.isin

    # We'll create index arrays for training and validation - NOT USING THESE
    # train_indices = []
    # val_indices = []

    for start in range(0, n_pairs, chunk_size):
        end = min(start + chunk_size, n_pairs)
        s1_chunk = pair_s1[start:end]
        t_chunk = pair_t[start:end]

        # Check which S1s in this chunk belong to train/val using lookup arrays
        in_train = train_s1_lookup[s1_chunk]
        in_val = val_s1_lookup[s1_chunk]

        n_train += np.sum(in_train)
        n_val += np.sum(in_val)

        # Store indices (relative to chunk start) - NOT USING THESE
        # train_indices.extend(np.where(in_train)[0])
        # val_indices.extend(np.where(in_val)[0])

    print(f"  Training pairs: {n_train:,}, Validation pairs: {n_val:,}",
          file=sys.stderr)

    # Second pass: allocate memmaps and fill with features and labels
    print("  Creating feature memmaps...", file=sys.stderr)

    # Define paths for training and validation feature directories
    train_feat_dir = f"{cache_dir}/feat_train"
    val_feat_dir = f"{cache_dir}/feat_test"
    os.makedirs(train_feat_dir, exist_ok=True)
    os.makedirs(val_feat_dir, exist_ok=True)

    # Function to check if memmap exists and has correct shape
    def check_and_load_memmap(filepath, dtype, shape, read_only=False):
        if os.path.exists(filepath):
            try:
                # Try to open in read-only mode to check shape
                existing = np.lib.format.open_memmap(filepath, mode='r', dtype=dtype, shape=shape)
                if existing.shape == shape:
                    if read_only:
                        return existing
                    else:
                        # If we need to write, we can still use it if we open in read-write mode
                        # But note: if the file exists and is correct, we can use it in read-write mode
                        return np.lib.format.open_memmap(filepath, mode='r+', dtype=dtype, shape=shape)
                else:
                    print(f"    Shape mismatch for {filepath}. Expected {shape}, got {existing.shape}. Recreating.", file=sys.stderr)
            except Exception as e:
                print(f"    Error reading {filepath}: {e}. Recreating.", file=sys.stderr)
        # If we get here, we need to create a new memmap
        return np.lib.format.open_memmap(filepath, mode='w+', dtype=dtype, shape=shape)

    # Check for existing training memmaps
    train_files = [
        ('X_train.npy', X.dtype, (n_train, X.shape[1])),
        ('pair_s1_train.npy', np.int64, (n_train,)),
        ('pair_t_train.npy', np.int64, (n_train,)),
        ('y_train.npy', np.int8, (n_train,))
    ]

    # Check for existing validation memmaps in feat_test directory
    val_files_feat_test = [
        ('X_test.npy', X.dtype, (n_val, X.shape[1])),
        ('pair_s1_test.npy', np.int64, (n_val,)),
        ('pair_t_test.npy', np.int64, (n_val,)),
        ('y_test.npy', np.int8, (n_val,))
    ]

    # Check for existing validation memmaps in feat_train directory (with _val suffix)
    val_files_feat_train = [
        ('X_val.npy', X.dtype, (n_val, X.shape[1])),
        ('pair_s1_val.npy', np.int64, (n_val,)),
        ('pair_t_val.npy', np.int64, (n_val,)),
        ('y_val.npy', np.int8, (n_val,))
    ]

    # Try to load training memmaps
    train_loaded = True
    try:
        dtrain_X = check_and_load_memmap(os.path.join(train_feat_dir, 'X_train.npy'), *train_files[0])
        dtrain_s1 = check_and_load_memmap(os.path.join(train_feat_dir, 'pair_s1_train.npy'), *train_files[1])
        dtrain_t = check_and_load_memmap(os.path.join(train_feat_dir, 'pair_t_train.npy'), *train_files[2])
        dtrain_y = check_and_load_memmap(os.path.join(train_feat_dir, 'y_train.npy'), *train_files[3])
        # Verify that all were loaded successfully (not recreated)
        # We can't easily check if they were recreated, but we assume if they exist and shape matches, they are loaded.
        print("    Using existing training memmaps.", file=sys.stderr)
    except Exception as e:
        print(f"    Failed to load training memmaps: {e}. Creating new ones.", file=sys.stderr)
        train_loaded = False

    if not train_loaded:
        # Create new training memmaps
        dtrain_X = np.lib.format.open_memmap(f"{train_feat_dir}/X_train.npy", mode='w+',
                                             dtype=X.dtype, shape=(n_train, X.shape[1]))
        dtrain_s1 = np.lib.format.open_memmap(f"{train_feat_dir}/pair_s1_train.npy", mode='w+',
                                              dtype=np.int64, shape=(n_train,))
        dtrain_t = np.lib.format.open_memmap(f"{train_feat_dir}/pair_t_train.npy", mode='w+',
                                             dtype=np.int64, shape=(n_train,))
        dtrain_y = np.lib.format.open_memmap(f"{train_feat_dir}/y_train.npy", mode='w+',
                                             dtype=np.int8, shape=(n_train,))
        print("    Created new training memmaps.", file=sys.stderr)

    # Try to load validation memmaps from feat_test directory first
    val_loaded = True
    try:
        dval_X = check_and_load_memmap(os.path.join(val_feat_dir, 'X_test.npy'), *val_files_feat_test[0])
        dval_s1 = check_and_load_memmap(os.path.join(val_feat_dir, 'pair_s1_test.npy'), *val_files_feat_test[1])
        dval_t = check_and_load_memmap(os.path.join(val_feat_dir, 'pair_t_test.npy'), *val_files_feat_test[2])
        dval_y = check_and_load_memmap(os.path.join(val_feat_dir, 'y_test.npy'), *val_files_feat_test[3])
        print("    Using existing validation memmaps from feat_test directory.", file=sys.stderr)
    except Exception as e:
        print(f"    Failed to load validation memmaps from feat_test: {e}. Trying feat_train directory.", file=sys.stderr)
        val_loaded = False

    if not val_loaded:
        # Try to load validation memmaps from feat_train directory (with _val suffix)
        try:
            dval_X = check_and_load_memmap(os.path.join(train_feat_dir, 'X_val.npy'), *val_files_feat_train[0])
            dval_s1 = check_and_load_memmap(os.path.join(train_feat_dir, 'pair_s1_val.npy'), *val_files_feat_train[1])
            dval_t = check_and_load_memmap(os.path.join(train_feat_dir, 'pair_t_val.npy'), *val_files_feat_train[2])
            dval_y = check_and_load_memmap(os.path.join(train_feat_dir, 'y_val.npy'), *val_files_feat_train[3])
            print("    Using existing validation memmaps from feat_train directory (with _val suffix).", file=sys.stderr)
        except Exception as e:
            print(f"    Failed to load validation memmaps from feat_train: {e}. Creating new ones in feat_test directory.", file=sys.stderr)
            val_loaded = False

    if not val_loaded:
        # Create new validation memmaps in feat_test directory
        dval_X = np.lib.format.open_memmap(f"{val_feat_dir}/X_test.npy", mode='w+',
                                           dtype=X.dtype, shape=(n_val, X.shape[1]))
        dval_s1 = np.lib.format.open_memmap(f"{val_feat_dir}/pair_s1_test.npy", mode='w+',
                                            dtype=np.int64, shape=(n_val,))
        dval_t = np.lib.format.open_memmap(f"{val_feat_dir}/pair_t_test.npy", mode='w+',
                                           dtype=np.int64, shape=(n_val,))
        dval_y = np.lib.format.open_memmap(f"{val_feat_dir}/y_test.npy", mode='w+',
                                           dtype=np.int8, shape=(n_val,))
        print("    Created new validation memmaps in feat_test directory.", file=sys.stderr)

    # Track any leftover data from previous chunk that didn't fit in allocated space
    leftover_train_X = None
    leftover_train_s1 = None
    leftover_train_t = None
    leftover_train_y = None
    leftover_val_X = None
    leftover_val_s1 = None
    leftover_val_t = None
    leftover_val_y = None

    if not (train_loaded and val_loaded):
        train_idx = 0
        val_idx = 0
        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            X_chunk = X[start:end]
            s1_chunk = pair_s1[start:end]
            t_chunk = pair_t[start:end]

            # Handle leftover from previous iteration
            if leftover_train_X is not None and leftover_train_X.shape[0] > 0:
                # Prepend leftover data to current chunk
                X_chunk = np.concatenate([leftover_train_X, X_chunk])
                s1_chunk = np.concatenate([leftover_train_s1, s1_chunk])
                t_chunk = np.concatenate([leftover_train_t, t_chunk])
                leftover_train_X = None
                leftover_train_s1 = None
                leftover_train_t = None
                leftover_train_y = None

            if leftover_val_X is not None and leftover_val_X.shape[0] > 0:
                # Prepend leftover data to current chunk
                X_chunk = np.concatenate([leftover_val_X, X_chunk])
                s1_chunk = np.concatenate([leftover_val_s1, s1_chunk])
                t_chunk = np.concatenate([leftover_val_t, t_chunk])
                leftover_val_X = None
                leftover_val_s1 = None
                leftover_val_t = None
                leftover_val_y = None

            # Check which S1s in this chunk belong to train/val using lookup arrays
            in_train = train_s1_lookup[s1_chunk]
            in_val = val_s1_lookup[s1_chunk]

            # Training data
            train_X_chunk = X_chunk[in_train]
            train_s1_chunk = s1_chunk[in_train]
            train_t_chunk = t_chunk[in_train]
            # For training labels, we look up in truth: 1 if (s1, t) in truth, else 0
            train_keys = (train_t_chunk.astype(np.int64) << np.int64(32)) | train_s1_chunk.astype(np.int64)
            train_pos_mask = np.isin(train_keys, keys_truth, assume_unique=True)

            # Validation data
            val_X_chunk = X_chunk[in_val]
            val_s1_chunk = s1_chunk[in_val]
            val_t_chunk = t_chunk[in_val]
            # For validation labels
            val_keys = (val_t_chunk.astype(np.int64) << np.int64(32)) | val_s1_chunk.astype(np.int64)
            val_pos_mask = np.isin(val_keys, keys_truth, assume_unique=True)

            # Handle training data
            if len(train_X_chunk) > 0:
                # Available space in training memmaps
                available_space = n_train - train_idx
                if len(train_X_chunk) > available_space:
                    # Not enough space - split the data
                    # Copy what fits
                    dtrain_X[train_idx:train_idx+available_space] = train_X_chunk[:available_space]
                    dtrain_s1[train_idx:train_idx+available_space] = train_s1_chunk[:available_space]
                    dtrain_t[train_idx:train_idx+available_space] = train_t_chunk[:available_space]
                    dtrain_y[train_idx:train_idx+available_space] = train_pos_mask[:available_space]

                    # Save leftover for next iteration
                    leftover_train_X = train_X_chunk[available_space:]
                    leftover_train_s1 = train_s1_chunk[available_space:]
                    leftover_train_t = train_t_chunk[available_space:]
                    leftover_train_y = train_pos_mask[available_space:]
                else:
                    # Enough space - copy all data
                    dtrain_X[train_idx:train_idx+len(train_X_chunk)] = train_X_chunk
                    dtrain_s1[train_idx:train_idx+len(train_X_chunk)] = train_s1_chunk
                    dtrain_t[train_idx:train_idx+len(train_X_chunk)] = train_t_chunk
                    dtrain_y[train_idx:train_idx+len(train_X_chunk)] = train_pos_mask
                    # No leftover

            # Handle validation data
            if len(val_X_chunk) > 0:
                # Available space in validation memmaps
                available_space = n_val - val_idx
                if len(val_X_chunk) > available_space:
                    # Not enough space - split the data
                    # Copy what fits
                    dval_X[val_idx:val_idx+available_space] = val_X_chunk[:available_space]
                    dval_s1[val_idx:val_idx+available_space] = val_s1_chunk[:available_space]
                    dval_t[val_idx:val_idx+available_space] = val_t_chunk[:available_space]
                    dval_y[val_idx:val_idx+available_space] = val_pos_mask[:available_space]

                    # Save leftover for next iteration
                    leftover_val_X = val_X_chunk[available_space:]
                    leftover_val_s1 = val_s1_chunk[available_space:]
                    leftover_val_t = val_t_chunk[available_space:]
                    leftover_val_y = val_pos_mask[available_space:]
                else:
                    # Enough space - copy all data
                    dval_X[val_idx:val_idx+len(val_X_chunk)] = val_X_chunk
                    dval_s1[val_idx:val_idx+len(val_X_chunk)] = val_s1_chunk
                    dval_t[val_idx:val_idx+len(val_X_chunk)] = val_t_chunk
                    dval_y[val_idx:val_idx+len(val_X_chunk)] = val_pos_mask
                    # No leftover

            train_idx += len(train_X_chunk)
            val_idx += len(val_X_chunk)

    print(f"  Training positives: {dtrain_y.sum():,} ({dtrain_y.mean()*100:.2f}%)",
          file=sys.stderr)
    print(f"  Validation positives: {dval_y.sum():,} ({dval_y.mean()*100:.2f}%)",
          file=sys.stderr)

    # LightGBM training
    print("  Training LightGBM model...", file=sys.stderr)
    try:
        import lightgbm as lgb
    except ImportError:
        print("  ERROR: LightGBM not installed. Install with: pip install lightgbm", file=sys.stderr)
        return

    # Create LightGBM datasets
    lgb_train = lgb.Dataset(dtrain_X, label=dtrain_y)
    lgb_val = lgb.Dataset(dval_X, label=dval_y, reference=lgb_train)

    # Set parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'is_unbalance': True,  # because we have few positives
        'seed': 42
    }

    # Train the model
    print("  Starting training...", file=sys.stderr)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_train, lgb_val],
                    callbacks=[lgb.early_stopping(stopping_rounds=50),
                               lgb.log_evaluation(100)])

    # Save the model
    print(f"  Saving model to {model_dir}...", file=sys.stderr)
    os.makedirs(model_dir, exist_ok=True)
    gbm.save_model(os.path.join(model_dir, "lgb_model.txt"))

    # Also save the feature metadata for later use in inference
    meta = {
        'feature_dim': X.shape[1],
        'train_pairs': n_train,
        'val_pairs': n_val,
        'unique_s1_count': len(all_s1),
        'max_s1_id': int(max_s1_id)
    }
    import json
    with open(os.path.join(model_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print("  Training completed successfully!", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--truth_dir", type=str, required=True)
    parser.add_argument("--ground_truth_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.feature_dir, args.truth_dir, args.ground_truth_path, args.cache_dir, args.model_dir)