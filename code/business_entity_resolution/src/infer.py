"""
Stage 8 - inference on test candidates and output generation.

Loads candidates, applies threshold via LightGBM model, writes TSVs.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb


def run(blob_dir: str, cand_dir: str, model_dir: str, out_dir: str) -> None:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    split = "test"

    print("* Loading model and threshold...", file=sys.stderr)
    bst = lgb.Booster(model_file=f"{model_dir}/model.txt")
    with open(f"{model_dir}/threshold.json", encoding="utf-8") as f:
        thr_data = json.load(f)
        thr = thr_data["threshold"]
    print(f"  threshold={thr:.3f}", file=sys.stderr)

    print("* Loading test string IDs from dataset/test/ ...", file=sys.stderr)
    s1_ids = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id"])["entity_id"].tolist()
    s2_ids = pd.read_csv("dataset/test/test_source2.tsv", sep="\t", usecols=["entity_id"])["entity_id"].tolist()
    s3_ids = pd.read_csv("dataset/test/test_source3.tsv", sep="\t", usecols=["entity_id"])["entity_id"].tolist()
    n2 = len(s2_ids)

    print("* Loading candidates and features...", file=sys.stderr)
    flat = np.load(f"{cand_dir}/cand.flat.npy")
    offs = np.load(f"{cand_dir}/cand.offs.npy")
    
    feat_dir = f"work/feat_{split}"
    if not os.path.exists(f"{feat_dir}/X.npy"):
        print(f"  ERROR: features not found in {feat_dir}", file=sys.stderr)
        sys.exit(1)
        
    X = np.load(f"{feat_dir}/X.npy", mmap_mode="r")
    pair_s1 = np.load(f"{feat_dir}/pair_s1.npy")
    pair_t = np.load(f"{feat_dir}/pair_t.npy")
    
    print(f"* Predicting on {len(X):,} pairs...", file=sys.stderr)
    preds = bst.predict(X)
    keep = preds >= thr
    
    cand_out_path = f"{out_dir}/candidate_pairs.tsv"
    match_out_path = f"{out_dir}/matching_results.tsv"
    
    print("* Formatting outputs...", file=sys.stderr)
    t_str_arr = np.array(s2_ids + s3_ids)
    
    # map candidate target integer IDs to strings
    cand_str_ids = t_str_arr[flat]
    
    # filter for final matches
    match_flat = flat[keep]
    match_s1 = pair_s1[keep]
    match_str_ids = t_str_arr[match_flat]
    
    # candidates grouped counts
    cand_counts = np.diff(offs)
    
    # matches grouped counts (pair_s1 is sequential from build_features chunking)
    match_counts = np.bincount(match_s1, minlength=len(s1_ids))
    
    print(f"* Writing {cand_out_path} and {match_out_path}...", file=sys.stderr)
    with open(cand_out_path, "w", encoding="utf-8") as fcand, \
         open(match_out_path, "w", encoding="utf-8") as fmatch:
         
        fcand.write("source1_entity_id\tcandidate_entity_ids\n")
        fmatch.write("source1_entity_id\tmatched_entity_ids\n")
        
        cand_idx = 0
        match_idx = 0
        
        for i in range(len(s1_ids)):
            s1 = s1_ids[i]
            
            # Write candidates
            cc = cand_counts[i]
            if cc > 0:
                c_ids = cand_str_ids[cand_idx:cand_idx+cc]
                fcand.write(f"{s1}\t{','.join(c_ids)}\n")
                cand_idx += cc
            else:
                fcand.write(f"{s1}\t\n")
                
            # Write matches
            mc = match_counts[i]
            if mc > 0:
                m_ids = match_str_ids[match_idx:match_idx+mc]
                fmatch.write(f"{s1}\t{','.join(m_ids)}\n")
                match_idx += mc
            else:
                fmatch.write(f"{s1}\t\n")
                
    print(f"* Inference complete ({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2], a[3], a[4])
