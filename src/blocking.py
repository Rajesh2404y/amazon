import re
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from src.normalization import normalize_business_name, normalize_business_address
from src.memory_utils import log_memory, force_garbage_collection

logger = logging.getLogger("entity_resolution")

RE_DBA = re.compile(r"\b(?:dba|d/b/a|ta|t/a)\s+(.+)$", re.IGNORECASE)

def extract_addr_blocking_keys(addr_str: Optional[str]) -> Set[str]:
    """
    Extracts high-precision address blocking keys:
    1. Number + adjacent word pairs (anywhere in address, stripped leading zeros)
    2. Street + city token pairs (number-independent)
    """
    if not addr_str or not isinstance(addr_str, str):
        return set()
    
    clean = re.sub(r"[^\w\s]", " ", addr_str.lower())
    words = clean.split()
    keys = set()
    
    # 1. Number + adjacent word pairs
    for i, w in enumerate(words):
        num = w.lstrip("0")
        if num.isdigit() and len(num) >= 1:
            if i + 1 < len(words) and not words[i+1].isdigit() and len(words[i+1]) >= 3:
                keys.add(f"{num}_{words[i+1]}")
            if i > 0 and not words[i-1].isdigit() and len(words[i-1]) >= 3:
                keys.add(f"{num}_{words[i-1]}")

    # 2. Significant non-numeric word pairs (first 3 words)
    non_nums = [w for w in words if not w.isdigit() and len(w) >= 3]
    if len(non_nums) >= 2:
        keys.add(f"{non_nums[0]}_{non_nums[1]}")
        
    return keys

def extract_char_3grams(text: Optional[str]) -> List[str]:
    """Extracts character 3-grams from normalized text."""
    if not text:
        return []
    clean = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(clean) >= 3:
        return [clean[i:i+3] for i in range(len(clean)-2)]
    return [clean] if clean else []

class MultiViewBlockingIndex:
    """
    Option A Multi-View Blocking Engine:
    Each view receives its own independent candidate quota BEFORE the union.
    No post-union truncation is applied, preventing true matches from being displaced.
    """
    def __init__(
        self,
        country: str,
        max_postings_per_key: int = 3000,
        name_top_k: int = 25,
        address_top_k: int = 15,
        char_top_k: int = 10
    ):
        self.country = country
        self.max_postings_per_key = max_postings_per_key
        self.name_top_k = name_top_k
        self.address_top_k = address_top_k
        self.char_top_k = char_top_k

        # Target records storage: (entity_id, name_norm, name_compact, name_tokens, addr_norm, addr_tokens, addr_numbers)
        self.targets: List[Tuple[str, str, str, Tuple[str, ...], str, Tuple[str, ...], Tuple[str, ...]]] = []
        
        # Inverted indices
        self.index_exact_name: Dict[str, List[int]] = defaultdict(list)
        self.index_pfx7: Dict[str, List[int]] = defaultdict(list)
        self.index_pfx5: Dict[str, List[int]] = defaultdict(list)
        self.index_two_tok: Dict[str, List[int]] = defaultdict(list)
        self.index_addr_key: Dict[str, List[int]] = defaultdict(list)
        self.index_char3: Dict[str, List[int]] = defaultdict(list)
        self.char_doc_freq: Dict[str, int] = defaultdict(int)

    def add_target_record(
        self,
        entity_id: str,
        name_norm: str,
        name_compact: str,
        name_tokens: Tuple[str, ...],
        addr_norm: str,
        addr_tokens: Tuple[str, ...],
        addr_numbers: Tuple[str, ...],
        raw_name: str = "",
        raw_addr: str = ""
    ):
        target_idx = len(self.targets)
        self.targets.append((
            entity_id, name_norm, name_compact, name_tokens, addr_norm, addr_tokens, addr_numbers
        ))

        # View 1: Exact Name Key & Prefixes
        if name_compact:
            if len(self.index_exact_name[name_compact]) < self.max_postings_per_key:
                self.index_exact_name[name_compact].append(target_idx)
            if len(name_compact) >= 7 and len(self.index_pfx7[name_compact[:7]]) < self.max_postings_per_key:
                self.index_pfx7[name_compact[:7]].append(target_idx)
            if len(name_compact) >= 5 and len(self.index_pfx5[name_compact[:5]]) < self.max_postings_per_key:
                self.index_pfx5[name_compact[:5]].append(target_idx)

        # View 2: Two-Token Prefix and Token Swap
        if len(name_tokens) >= 2:
            k1 = f"{name_tokens[0]}_{name_tokens[1]}"
            k2 = f"{name_tokens[1]}_{name_tokens[0]}"
            if len(self.index_two_tok[k1]) < self.max_postings_per_key:
                self.index_two_tok[k1].append(target_idx)
            if len(self.index_two_tok[k2]) < self.max_postings_per_key:
                self.index_two_tok[k2].append(target_idx)
        elif len(name_tokens) == 1:
            if len(self.index_two_tok[name_tokens[0]]) < self.max_postings_per_key:
                self.index_two_tok[name_tokens[0]].append(target_idx)

        # View 3: DBA Names
        check_text = raw_name or name_norm
        dba_m = RE_DBA.search(check_text)
        if dba_m:
            _, d_comp, _, _ = normalize_business_name(dba_m.group(1))
            if d_comp and len(self.index_exact_name[d_comp]) < self.max_postings_per_key:
                self.index_exact_name[d_comp].append(target_idx)
            if len(d_comp) >= 7 and len(self.index_pfx7[d_comp[:7]]) < self.max_postings_per_key:
                self.index_pfx7[d_comp[:7]].append(target_idx)

        # View 4: Address Keys (Number + Street, anywhere in address)
        addr_text = raw_addr or addr_norm
        if addr_text:
            for ak in extract_addr_blocking_keys(addr_text):
                if len(self.index_addr_key[ak]) < self.max_postings_per_key:
                    self.index_addr_key[ak].append(target_idx)

        # View 5: Character 3-Grams with doc frequency tracking
        if name_norm:
            seen_grams = set(extract_char_3grams(name_norm))
            for g in seen_grams:
                self.char_doc_freq[g] += 1
                if len(self.index_char3[g]) < 1500:
                    self.index_char3[g].append(target_idx)

    def query_s1_entity(
        self,
        name_norm: str,
        name_compact: str,
        name_tokens: Tuple[str, ...],
        addr_norm: str,
        addr_tokens: Tuple[str, ...],
        addr_numbers: Tuple[str, ...],
        raw_addr: str = "",
        name_top_k: Optional[int] = None,
        address_top_k: Optional[int] = None,
        char_top_k: Optional[int] = None
    ) -> List[Tuple[str, int]]:
        """
        OPTION A IMPLEMENTATION:
        1. Select Top-K candidates independently for Name view.
        2. Select Top-K candidates independently for Address view.
        3. Select Top-K candidates independently for Character view.
        4. Form the mathematical UNION of all selected candidates.
        5. DO NOT apply a post-union truncation.
        """
        k_name = name_top_k or self.name_top_k
        k_addr = address_top_k or self.address_top_k
        k_char = char_top_k or self.char_top_k

        # --- VIEW 1: NAME CANDIDATES ---
        name_hits = defaultdict(int)
        if name_compact and name_compact in self.index_exact_name:
            for tidx in self.index_exact_name[name_compact]:
                name_hits[tidx] += 10
        if len(name_compact) >= 7 and name_compact[:7] in self.index_pfx7:
            for tidx in self.index_pfx7[name_compact[:7]]:
                name_hits[tidx] += 6
        if len(name_tokens) >= 2:
            k = f"{name_tokens[0]}_{name_tokens[1]}"
            if k in self.index_two_tok:
                for tidx in self.index_two_tok[k]:
                    name_hits[tidx] += 4
        elif len(name_tokens) == 1 and name_tokens[0] in self.index_two_tok:
            for tidx in self.index_two_tok[name_tokens[0]]:
                name_hits[tidx] += 4
        if len(name_compact) >= 5 and name_compact[:5] in self.index_pfx5:
            for tidx in self.index_pfx5[name_compact[:5]]:
                name_hits[tidx] += 2

        top_name_tindices = {tidx for tidx, _ in sorted(name_hits.items(), key=lambda x: x[1], reverse=True)[:k_name]}

        # --- VIEW 2: ADDRESS CANDIDATES ---
        addr_hits = defaultdict(int)
        query_addr = raw_addr or addr_norm
        if query_addr:
            for ak in extract_addr_blocking_keys(query_addr):
                if ak in self.index_addr_key:
                    for tidx in self.index_addr_key[ak]:
                        addr_hits[tidx] += 8

        top_addr_tindices = {tidx for tidx, _ in sorted(addr_hits.items(), key=lambda x: x[1], reverse=True)[:k_addr]}

        # --- VIEW 3: CHARACTER 3-GRAM CANDIDATES ---
        char_hits = defaultdict(float)
        if name_norm:
            s_grams = extract_char_3grams(name_norm)
            for g in s_grams:
                df = self.char_doc_freq.get(g, 1)
                idf = 1.0 / (1.0 + (df / 500.0))
                for tidx in self.index_char3.get(g, []):
                    char_hits[tidx] += idf

        top_char_tindices = {tidx for tidx, _ in sorted(char_hits.items(), key=lambda x: x[1], reverse=True)[:k_char]}

        # --- OPTION A: MATHEMATICAL UNION OF INDEPENDENT VIEWS ---
        union_tindices = top_name_tindices | top_addr_tindices | top_char_tindices

        if not union_tindices:
            return []

        # Track multi-view rule agreement count
        results = []
        for tidx in union_tindices:
            cid = self.targets[tidx][0]
            views_hit = 0
            if tidx in top_name_tindices: views_hit += 1
            if tidx in top_addr_tindices: views_hit += 1
            if tidx in top_char_tindices: views_hit += 1
            results.append((cid, views_hit))

        # Sort descending by rule hits for consistent ordering
        results.sort(key=lambda x: x[1], reverse=True)
        return results
