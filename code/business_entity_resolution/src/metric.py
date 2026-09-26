"""
Stage 4 - train the pair classifier (LightGBM) and calibrate the threshold.

Training data: candidate pairs of a *validation* split of train S1 records.
The model never trains on the S1 entities it is evaluated on.

Inputs (feature_dir): X.npy (N x NF float32 memmap), pair_s1.npy, pair_t.npy,
pos_s1.npy / pos_t.npy (true pairs among candidates).

Outputs: model.txt (LightGBM booster), threshold.json (F0.5-optimal global
threshold + per-record top-1 policy check), metrics.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def f05_macro_from_preds(s1_rows, pred, truth_by_s1, n_s1, thr):
    """Macro F0.5 over S1 entities with global threshold ``thr``.

    pred: float32 score per pair (aligned with s1_rows order)
    truth_by_s1: dict s1_row -> set(target idx)
    """
    keep = pred >= thr
    # group predicted targets per s1 row
    order = np.argsort(s1_rows, kind="stable")
    sk = s1_rows[order]
    pk = keep[order]
    tk = _targets_sorted[order] if False else None
    # For memory reasons we compute per-entity precision via np ops below.
    return None


def f05_of_entity(pred_set: set, true_set: set) -> float:
    """F0.5 of one S1 entity (singleton handled naturally)."""
    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    prec = tp / len(pred_set)
    rec = tp / len(true_set)
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def evaluate_groups(s1_rows: np.ndarray, targets: np.ndarray, pred: np.ndarray,
                    truth: dict, n_s1: int, thr: float) -> tuple[float, dict]:
    """Macro F0.5 with predicted set = {targets with score >= thr} per S1."""
    keep = pred >= thr
    order = np.argsort(s1_rows[keep], kind="stable") if keep.any() else slice(0, 0)
    # build per-s1 predicted sets in one pass (python loop over kept pairs)
    pred_sets = {}
    s1k = s1_rows[keep]
    tk = targets[keep]
    for i in range(len(s1k)):
        s = s1k[i]
        st = pred_sets.get(s)
        if st is None:
            pred_sets[s] = {tk[i]}
        else:
            st.add(tk[i])
    # score every S1 that appears in truth or has predictions
    total = 0.0
    n = 0
    err_stats = {"fp_only": 0, "missed": 0, "perfect": 0}
    for s in range(n_s1):
        true_set = truth.get(s)
        pset = pred_sets.get(s, set())
        if true_set is None and not pset:
            total += 1.0      # true singleton, predicted singleton
            n += 1
            err_stats["perfect"] += 1
            continue
        tset = true_set if true_set is not None else set()
        total += f05_of_entity(pset, tset)
        n += 1
        if not pset and tset:
            err_stats["missed"] += 1
        elif pset and not tset:
            err_stats["fp_only"] += 1
        elif pset == tset:
            err_stats["perfect"] += 1
    return total / max(n, 1), err_stats
