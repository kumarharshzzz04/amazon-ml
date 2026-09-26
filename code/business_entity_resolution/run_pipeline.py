"""
One-command pipeline driver with resume support.

Each stage is skipped if its output already exists, so you can stop/restart
or run stages in parallel across machines (one stage per machine).

Usage (from repo root):
  python code/business_entity_resolution/run_pipeline.py all
  python code/business_entity_resolution/run_pipeline.py prep
  python code/business_entity_resolution/run_pipeline.py keys
  python code/business_entity_resolution/run_pipeline.py candidates <split> [k thr top_m]
  python code/business_entity_resolution/run_pipeline.py recall
  python code/business_entity_resolution/run_pipeline.py blobs
  python code/business_entity_resolution/run_pipeline.py features [split]
  python code/business_entity_resolution/run_pipeline.py train
  python code/business_entity_resolution/run_pipeline.py inference
"""
from __future__ import annotations

import os
import subprocess
import sys

PY = sys.executable
CODE = os.path.dirname(os.path.abspath(__file__))          # .../business_entity_resolution
ROOT = os.path.dirname(os.path.dirname(CODE))              # repo root
SRC = os.path.join(CODE, "src")
ENV = dict(os.environ, PYTHONPATH=CODE, PYTHONIOENCODING="utf-8")


def sh(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    r = subprocess.run(args, cwd=ROOT, env=ENV)
    if r.returncode != 0:
        sys.exit(f"stage failed: {' '.join(args)}")


def stage_prep():
    sh([PY, "-m", "src.prep",
        "student_resource/student_resource/dataset", "work/cache"])


def stage_keys():
    sh([PY, "-m", "src.keys", "work/cache", "work/keys2"])


def stage_candidates(split: str, k=10, thr=2000.0, top_m=150):
    out = f"work/cand_{split}"
    if os.path.exists(f"{out}/cand.count.npy"):
        print(f"[skip] {out} already complete")
        return
    sh([PY, "-m", "src.candidates", "work/keys2", out, split,
        str(k), str(thr), str(top_m)])


def stage_recall():
    if os.path.exists("work/cand_train/pos_s1.npy"):
        print("[skip] recall already done")
        return
    sh([PY, "-m", "src.eval_recall", "work/cache", "work/cand_train",
        "work/cand_train",
        "student_resource/student_resource/dataset/train/train_ground_truth.tsv",
        "train"])


def stage_blobs():
    sh([PY, "-m", "src.blobs", "work/cache", "work/keys2", "work/blobs"])


def stage_features(split: str, workers: int = 6):
    out = f"work/feat_{split}"
    if os.path.exists(f"{out}/X.npy"):
        print(f"[skip] {out} exists")
        return
    sh([PY, "-m", "src.build_features", "work/blobs", f"work/cand_{split}",
        out, split, str(workers)])


def stage_train():
    if os.path.exists("work/model/model.txt"):
        print("[skip] model exists")
        return
    sh([PY, "-m", "src.train", "work/feat_train", "work/cand_train",
        "student_resource/student_resource/dataset/train/train_ground_truth.tsv",
        "work/cache", "work/model"])


def stage_inference():
    sh([PY, "-m", "src.infer", "work/blobs", "work/cand_test", "work/model",
        "output"])


STAGES = {
    "prep": stage_prep,
    "keys": stage_keys,
    "candidates": lambda: (stage_candidates("train"), stage_candidates("test")),
    "recall": stage_recall,
    "blobs": stage_blobs,
    "features": lambda: (stage_features("train"), stage_features("test")),
    "train": stage_train,
    "inference": stage_inference,
}

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "all":
        for name, fn in STAGES.items():
            print(f"\n===== STAGE {name} =====", flush=True)
            fn()
    else:
        STAGES[what]()
