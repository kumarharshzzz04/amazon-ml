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
    truth = build_truth_rows(ground_truth_path)
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
    feature_dir = f"{cache_dir}/feat_train"
    os.makedirs(feature_dir, exist_ok=True)
    dtrain_X = np.lib.format.open_memmap(f"{feature_dir}/X_train.npy", mode='w+',
                                         dtype=X.dtype, shape=(n_train,))
    dtrain_s1 = np.lib.format.open_memmap(f"{feature_dir}/pair_s1_train.npy", mode='w+',
                                          dtype=np.int64, shape=(n_train,))
    dtrain_t = np.lib.format.open_memmap(f"{feature_dir}/pair_t_train.npy", mode='w+',
                                         dtype=np.int64, shape=(n_train,))
    dtrain_y = np.lib.format.open_memmap(f"{feature_dir}/y_train.npy", mode='w+',
                                         dtype=np.int8, shape=(n_train,))

    feature_dir = f"{cache_dir}/feat_test"
    os.makedirs(feature_dir, exist_ok=True)
    dval_X = np.lib.format.open_memmap(f"{feature_dir}/X_test.npy", mode='w+',
                                       dtype=X.dtype, shape=(n_val,))
    dval_s1 = np.lib.format.open_memmap(f"{feature_dir}/pair_s1_test.npy", mode='w+',
                                        dtype=np.int64, shape=(n_val,))
    dval_t = np.lib.format.open_memmap(f"{feature_dir}/pair_t_test.npy", mode='w+',
                                       dtype=np.int64, shape=(n_val,))
    dval_y = np.lib.format.open_memmap(f"{feature_dir}/y_test.npy", mode='w+',
                                       dtype=np.int8, shape=(n_val,))

    # Track any leftover data from previous chunk that didn't fit in allocated space
    leftover_train_X = None
    leftover_train_s1 = None
    leftover_train_t = None
    leftover_train_y = None
    leftover_val_X = None
    leftover_val_s1 = None
    leftover_val_t = None
    leftover_val_y = None

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

        # Compute labels for what fits
        train_keys_pairs = (t_chunk[:available_space].astype(np.int64) << np.int64(32)) | s1_chunk[:available_space].astype(np.int64)
        val_keys_pairs = (t_chunk[:available_space].astype(np.int64) << np.int64(32)) | s1_chunk[:available_space].astype(np.int64)
        train_pos_mask = np.isin(train_keys_pairs, keys_truth, assume_unique=True)
        val_pos_mask = np.isin(val_keys_pairs, keys_truth, assume_unique=True)

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

        train_idx += available_space  # We've filled available_space more elements
        val_idx += available_space  # We've filled available_space more elements

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

    # Training parameters
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    # Train model
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=[lgb_train, lgb_val],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(period=50)
                    ])

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/model.txt"
    gbm.save_model(model_path)
    print(f"  Model saved to {model_path}", file=sys.stderr)

    # Feature importance
    print("  Top 20 features by importance:", file=sys.stderr)
    for i, (imp, name) in enumerate(zip(gbm.feature_importance(importance_type='gain'),
                                        [f"f_{i}" for i in range(X.shape[1])])):
        if i >= 20:
            break
        print(f"    {i+1:2d}. {name}: {imp}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train business entity resolution model')
    parser.add_argument('feature_dir', help='Directory containing feature files')
    parser.add_argument('truth_dir', help='Directory containing truth files')
    parser.add_argument('ground_truth_path', help='Path to ground truth TSV file')
    parser.add_argument('cache_dir', help='Directory for intermediate cache files')
    parser.add_argument('model_dir', help='Directory to save trained model')
    args = parser.parse_args()
    main(args.feature_dir, args.truth_dir, args.ground_truth_path, args.cache_dir, args.model_dir)