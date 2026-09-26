import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from rapidfuzz import fuzz, distance

FEATURE_NAMES = [
    # Name features
    "name_exact_match",
    "name_jaro_winkler",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_length_diff_ratio",
    "name_prefix_4_match",
    "name_token_jaccard",
    # Address features
    "addr_exact_match",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_number_overlap_ratio",
    "addr_length_diff_ratio",
    "addr_is_empty",
    # Joint & Metadata features
    "blocking_rule_hits",
    "candidate_is_s2",
    "name_len_s1",
    "addr_len_s1",
]

def jaccard_similarity(tokens1: Tuple[str, ...], tokens2: Tuple[str, ...]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    s1, s2 = set(tokens1), set(tokens2)
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return float(intersection / union) if union > 0 else 0.0

def number_overlap_ratio(nums1: Tuple[str, ...], nums2: Tuple[str, ...]) -> float:
    if not nums1 or not nums2:
        return 0.0
    s1, s2 = set(nums1), set(nums2)
    intersection = len(s1 & s2)
    min_len = min(len(s1), len(s2))
    return float(intersection / min_len) if min_len > 0 else 0.0

def extract_pair_features(
    s1_norm: str,
    s1_compact: str,
    s1_tokens: Tuple[str, ...],
    s1_anorm: str,
    s1_atokens: Tuple[str, ...],
    s1_anums: Tuple[str, ...],
    cand_id: str,
    cand_norm: str,
    cand_compact: str,
    cand_tokens: Tuple[str, ...],
    cand_anorm: str,
    cand_atokens: Tuple[str, ...],
    cand_anums: Tuple[str, ...],
    rule_hits: int = 1
) -> List[float]:
    """
    Extracts high-dimensional similarity features for a candidate pair using RapidFuzz.
    Optimized for CPU throughput and minimal object allocations.
    """
    # 1. Name Features
    name_exact = 1.0 if s1_compact and s1_compact == cand_compact else 0.0
    name_jw = distance.JaroWinkler.similarity(s1_norm, cand_norm)
    name_lev = fuzz.ratio(s1_norm, cand_norm) / 100.0
    name_sort = fuzz.token_sort_ratio(s1_norm, cand_norm) / 100.0
    name_set = fuzz.token_set_ratio(s1_norm, cand_norm) / 100.0
    name_partial = fuzz.partial_ratio(s1_norm, cand_norm) / 100.0
    
    len_diff = abs(len(s1_norm) - len(cand_norm))
    max_len = max(len(s1_norm), len(cand_norm), 1)
    name_len_diff = 1.0 - (len_diff / max_len)
    
    name_pfx4 = 1.0 if (len(s1_compact) >= 4 and len(cand_compact) >= 4 and s1_compact[:4] == cand_compact[:4]) else 0.0
    name_jaccard = jaccard_similarity(s1_tokens, cand_tokens)

    # 2. Address Features
    addr_empty = 1.0 if (not s1_anorm or not cand_anorm) else 0.0
    if addr_empty == 1.0:
        addr_exact = 0.0
        addr_sort = 0.0
        addr_set = 0.0
        addr_num_overlap = 0.0
        addr_len_diff = 0.0
    else:
        addr_exact = 1.0 if s1_anorm == cand_anorm else 0.0
        addr_sort = fuzz.token_sort_ratio(s1_anorm, cand_anorm) / 100.0
        addr_set = fuzz.token_set_ratio(s1_anorm, cand_anorm) / 100.0
        addr_num_overlap = number_overlap_ratio(s1_anums, cand_anums)
        alen_diff = abs(len(s1_anorm) - len(cand_anorm))
        amax_len = max(len(s1_anorm), len(cand_anorm), 1)
        addr_len_diff = 1.0 - (alen_diff / amax_len)

    # 3. Joint & Metadata Features
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    
    return [
        name_exact,
        name_jw,
        name_lev,
        name_sort,
        name_set,
        name_partial,
        name_len_diff,
        name_pfx4,
        name_jaccard,
        addr_exact,
        addr_sort,
        addr_set,
        addr_num_overlap,
        addr_len_diff,
        addr_empty,
        float(rule_hits),
        is_s2,
        float(len(s1_norm)),
        float(len(s1_anorm))
    ]
