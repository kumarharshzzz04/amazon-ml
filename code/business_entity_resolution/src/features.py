"""
Stage 3 - pair feature engineering.

Features for each (S1, target) candidate pair, computed with rapidfuzz on
normalized text plus token-set algebra.  All features describe agreement
between the two records (they are symmetric in construction).

Groups:
  name  - token jaccard/overlap, partial + set ratios, char-bigram cosine,
          legal-suffix agreement, length ratios
  addr  - token jaccard/overlap, partial + set ratios, digit-run agreement,
          postal agreement, state agreement, has-address flags
  meta  - country match, source-pair flags, idf shared-key score/count

Feature vector layout is fixed by FEAT_NAMES so train/inference always agree.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import normalize as N  # noqa: E402

FEAT_NAMES = [
    # name
    "n_jaccard", "n_overlap", "n_ratio", "n_partial", "n_token_set",
    "n_token_sort", "n_charbigram", "n_jw", "n_len_ratio", "n_suffix_agree",
    "n_suffix_both", "n_nontok", "n_core_jaccard",
    # address
    "a_jaccard", "a_overlap", "a_ratio", "a_partial", "a_token_set",
    "a_token_sort", "a_charbigram", "a_jw", "a_len_ratio",
    "a_digit_jaccard", "a_digit_contains", "a_postal_eq", "a_state_eq",
    "a_state_one_sided", "a_empty_one", "a_empty_both",
    # meta
    "country_eq", "key_score", "key_count", "key_count_ratio", "postal_hash_eq",
]
NF = len(FEAT_NAMES)

# NOTE: the canonical feature implementation is feats_fast.compute_chunk (blob
# based, used by build_features).  pair_features below is a slow reference
# implementation kept for unit checks; it does not emit postal_hash_eq.


def _charbigram_cos(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    def grams(s):
        return Counter(s[i:i+2] for i in range(len(s) - 1))
    ga, gb = grams(a), grams(b)
    dot = sum((ga & gb).values())
    na, nb = sum(ga.values()), sum(gb.values())
    if not na or not nb:
        return 0.0
    return dot / (na ** 0.5 * nb ** 0.5)


def _legal_profile(toks) -> tuple[set, bool]:
    suf = {t for t in toks if t in N.LEGAL_TOKENS}
    return suf, len(suf) > 0


def name_features(n1: str, n2: str):
    t1, t2 = set(N.name_tokens(n1)), set(N.name_tokens(n2))
    s1, s2 = N.norm_text(n1), N.norm_text(n2)
    jac = N.jaccard(t1, t2)
    ov = N.overlap_coef(t1, t2)
    suf1, b1 = _legal_profile(t1)
    suf2, b2 = _legal_profile(t2)
    core1 = t1 - suf1
    core2 = t2 - suf2
    core_jac = N.jaccard(core1, core2) if core1 or core2 else jac
    return [
        jac,
        ov,
        fuzz.ratio(s1, s2) / 100.0,
        fuzz.partial_ratio(s1, s2) / 100.0,
        fuzz.token_set_ratio(s1, s2) / 100.0,
        fuzz.token_sort_ratio(s1, s2) / 100.0,
        _charbigram_cos(s1, s2),
        JaroWinkler.similarity(s1, s2),
        min(len(s1), len(s2)) / max(len(s1), len(s2), 1),
        N.jaccard(suf1, suf2) if suf1 or suf2 else 1.0,
        1.0 if (b1 and b2) else 0.0,
        1.0 if (not t1 and not t2) else 0.0,
        # core-token jaccard folded into overlap slot ordering below
    ], core_jac


def addr_features(a1: str, a2: str):
    t1, t2 = set(N.addr_tokens(a1)), set(N.addr_tokens(a2))
    s1, s2 = N.norm_text(a1), N.norm_text(a2)
    d1, d2 = set(N.numeric_tokens(a1)), set(N.numeric_tokens(a2))
    e1, e2 = (not a1.strip()), (not a2.strip())
    st1 = {t for t in t1 if t in N.STATE_TOKENS and len(t) == 2}
    st2 = {t for t in t2 if t in N.STATE_TOKENS and len(t) == 2}
    p1 = {x for x in d1 if len(x) in (5, 6)}
    p2 = {x for x in d2 if len(x) in (5, 6)}
    return [
        N.jaccard(t1, t2),
        N.overlap_coef(t1, t2),
        fuzz.ratio(s1, s2) / 100.0 if s1 and s2 else (1.0 if e1 and e2 else 0.0),
        fuzz.partial_ratio(s1, s2) / 100.0 if s1 and s2 else (1.0 if e1 and e2 else 0.0),
        fuzz.token_set_ratio(s1, s2) / 100.0 if s1 and s2 else (1.0 if e1 and e2 else 0.0),
        fuzz.token_sort_ratio(s1, s2) / 100.0 if s1 and s2 else (1.0 if e1 and e2 else 0.0),
        _charbigram_cos(s1, s2),
        JaroWinkler.similarity(s1, s2) if s1 and s2 else (1.0 if e1 and e2 else 0.0),
        min(len(s1), len(s2)) / max(len(s1), len(s2), 1),
        N.jaccard(d1, d2),
        1.0 if (d1 and d2 and (d1 <= d2 or d2 <= d1)) else 0.0,
        1.0 if (p1 and p1 & p2) else 0.0,
        1.0 if (st1 and st1 & st2) else 0.0,
        1.0 if (st1 != st2) else 0.0,
        1.0 if (e1 != e2) else 0.0,
        1.0 if (e1 and e2) else 0.0,
    ]


def pair_features(rec1, rec2, key_score: float, key_count: int,
                  n_keys1: int, n_keys2: int) -> list[float]:
    """Full feature vector for one candidate pair.

    rec1/rec2: (name_norm, addr_norm, country) tuples.
    """
    n1, a1, c1 = rec1
    n2, a2, c2 = rec2
    nf, core_jac = name_features(n1, n2)
    af = addr_features(a1, a2)
    return nf[:12] + [core_jac] + af + [
        1.0 if c1 == c2 else 0.0,
        float(key_score),
        float(key_count),
        float(key_count) / max(n_keys1, n_keys2, 1),
    ]
