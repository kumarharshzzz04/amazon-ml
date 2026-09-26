"""
Synthetic correctness test for the candidate inverted index.

Catches the two bugs this pipeline once had:
  1. S3 records mapped into S2's row space (missing +n2 offset)
  2. per-source argsorts concatenated so postings were not globally
     ordered by key id (koff slices misaligned)

Run from repo root:
  python code/business_entity_resolution/tests/test_index.py
Prints OK per case and exits 0; any failure exits 1.
"""
import os
import sys
import tempfile

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

from src.candidates import build_index  # noqa: E402


def make_fake_keys(d: str, ids2, off2, ids3, off3) -> str:
    np.save(os.path.join(d, "train_source2.ids.npy"), np.asarray(ids2, dtype=np.int32))
    np.save(os.path.join(d, "train_source2.offs.npy"), np.asarray(off2, dtype=np.int64))
    np.save(os.path.join(d, "train_source3.ids.npy"), np.asarray(ids3, dtype=np.int32))
    np.save(os.path.join(d, "train_source3.offs.npy"), np.asarray(off3, dtype=np.int64))
    return d


def run_case(name, ids2, off2, ids3, off3, expectations):
    with tempfile.TemporaryDirectory() as d:
        idx = build_index(make_fake_keys(d, ids2, off2, ids3, off3), "train")
        koff, order = idx["koff"], idx["order"]
        failed = False
        for kid, expect in expectations.items():
            got = set(np.asarray(order[koff[kid]:koff[kid + 1]]).tolist())
            ok = got == set(expect)
            print(f"  {name} key {kid}: {'OK' if ok else f'BUG got {sorted(got)} expect {sorted(expect)}'}")
            failed |= not ok
        # df must equal posting counts everywhere
        counts_ok = all(
            koff[k + 1] - koff[k] == idx["df"][k] for k in expectations
        )
        print(f"  {name} df==postings: {'OK' if counts_ok else 'BUG'}")
        return not failed and counts_ok


def main() -> int:
    all_ok = True

    # Case 1: the exact bug pair — same key in both sources; s3 row must be n2-offset
    # (s2 rec0=[3,7], rec1=[3]; s3 rec0=[1], rec1=[3])
    all_ok &= run_case(
        "offset",
        ids2=[3, 7, 3], off2=[0, 2, 3],
        ids3=[1, 3], off3=[0, 1, 2],
        expectations={1: {2}, 3: {0, 1, 3}, 7: {0}},
    )

    # Case 2: interleave keys so separate argsorts would misorder postings
    all_ok &= run_case(
        "global-order",
        ids2=[9, 1], off2=[0, 1, 2],
        ids3=[1, 9], off3=[0, 1, 2],
        # global rows: s2 rec0=0 holds 9; s2 rec1=1 holds 1; s3 rec0=2 holds 1; s3 rec1=3 holds 9
        expectations={1: {1, 2}, 9: {0, 3}},
    )

    # Case 3: repeated keys inside one record collapse to one posting per record
    # (s2 rec0 emits keys [5,5,6] -> rows for 5:{0}, 6:{0}; s2 rec1 emits [6] -> 6:{1}; s3 rec0 [6] -> 6:{2})
    all_ok &= run_case(
        "duplicates",
        ids2=[5, 5, 6, 6], off2=[0, 3, 4],
        ids3=[6], off3=[0, 1],
        expectations={5: {0}, 6: {0, 1, 2}},
    )

    print("\nALL INDEX TESTS PASSED" if all_ok else "\nINDEX TESTS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
