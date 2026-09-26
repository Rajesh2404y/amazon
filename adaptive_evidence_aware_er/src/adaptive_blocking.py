import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from src.normalization import normalize_business_name, normalize_business_address
from src.blocking import extract_addr_blocking_keys, extract_char_3grams, RE_DBA

STOP_WORDS = {"the", "a", "an", "and", "of", "for", "in", "on", "at", "to", "by", "with", "co"}

def extract_significant_tokens(tokens: Tuple[str, ...]) -> List[str]:
    """Returns tokens excluding common stop words and length < 3."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) >= 3]

def get_sorted_token_key(tokens: Tuple[str, ...]) -> Optional[str]:
    """Returns sorted bigram of first two significant tokens."""
    sig = extract_significant_tokens(tokens)
    if len(sig) >= 2:
        pair = sorted([sig[0], sig[1]])
        return f"{pair[0]}_{pair[1]}"
    elif len(sig) == 1:
        return sig[0]
    return None

def get_compound_num_token_keys(name_tokens: Tuple[str, ...], addr_numbers: Tuple[str, ...]) -> Set[str]:
    """Returns compound (street_number, name_token) keys for ultra-high-precision recovery."""
    keys = set()
    sig = extract_significant_tokens(name_tokens)
    if not sig or not addr_numbers:
        return keys
    
    first_token = sig[0]
    for num in addr_numbers:
        clean_num = num.lstrip("0")
        if clean_num:
            keys.add(f"{clean_num}_{first_token}")
    return keys

class AdaptiveMultiViewBlockingIndex:
    """
    Evidence-Aware Adaptive Multi-View Blocking Index.
    Preserves strict per-view quotas before union, with evidence-based dynamic quota allocation.
    """
    def __init__(self, country: str, max_postings_per_key: int = 3000):
        self.country = country
        self.max_postings_per_key = max_postings_per_key

        self.targets: List[Tuple[str, str, str, Tuple[str, ...], str, Tuple[str, ...], Tuple[str, ...]]] = []
        
        # Existing Baseline Views
        self.index_exact_name: Dict[str, List[int]] = defaultdict(list)
        self.index_pfx7: Dict[str, List[int]] = defaultdict(list)
        self.index_pfx5: Dict[str, List[int]] = defaultdict(list)
        self.index_two_tok: Dict[str, List[int]] = defaultdict(list)
        self.index_addr_key: Dict[str, List[int]] = defaultdict(list)
        self.index_char3: Dict[str, List[int]] = defaultdict(list)
        self.char_doc_freq: Dict[str, int] = defaultdict(int)

        # New Controlled Recovery Views
        self.index_sorted_token: Dict[str, List[int]] = defaultdict(list)
        self.index_compound_num_name: Dict[str, List[int]] = defaultdict(list)

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

        # View 4: Address Keys (Number + Street)
        addr_text = raw_addr or addr_norm
        if addr_text:
            for ak in extract_addr_blocking_keys(addr_text):
                if len(self.index_addr_key[ak]) < self.max_postings_per_key:
                    self.index_addr_key[ak].append(target_idx)

        # View 5: Character 3-Grams
        if name_norm:
            seen_grams = set(extract_char_3grams(name_norm))
            for g in seen_grams:
                self.char_doc_freq[g] += 1
                if len(self.index_char3[g]) < 1500:
                    self.index_char3[g].append(target_idx)

        # View 6 (Recovery): Sorted Significant Token Key
        sort_k = get_sorted_token_key(name_tokens)
        if sort_k and len(self.index_sorted_token[sort_k]) < self.max_postings_per_key:
            self.index_sorted_token[sort_k].append(target_idx)

        # View 7 (Recovery): Compound Number + First Name Token
        comp_keys = get_compound_num_token_keys(name_tokens, addr_numbers)
        for ck in comp_keys:
            if len(self.index_compound_num_name[ck]) < self.max_postings_per_key:
                self.index_compound_num_name[ck].append(target_idx)

    def query_s1_adaptive(
        self,
        name_norm: str,
        name_compact: str,
        name_tokens: Tuple[str, ...],
        addr_norm: str,
        addr_tokens: Tuple[str, ...],
        addr_numbers: Tuple[str, ...],
        raw_addr: str = "",
        mode: str = "baseline", # "baseline", "adaptive_a", "adaptive_b", "adaptive_c", "adaptive_d"
        max_budget: int = 50
    ) -> List[Tuple[str, int]]:
        
        # 1. Collect Name Hits
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

        # 2. Collect Address Hits
        addr_hits = defaultdict(int)
        query_addr = raw_addr or addr_norm
        if query_addr:
            for ak in extract_addr_blocking_keys(query_addr):
                if ak in self.index_addr_key:
                    for tidx in self.index_addr_key[ak]:
                        addr_hits[tidx] += 8

        # 3. Collect Char Hits
        char_hits = defaultdict(float)
        if name_norm:
            s_grams = extract_char_3grams(name_norm)
            for g in s_grams:
                df = self.char_doc_freq.get(g, 1)
                idf = 1.0 / (1.0 + (df / 500.0))
                for tidx in self.index_char3.get(g, []):
                    char_hits[tidx] += idf

        # 4. Collect Recovery Hits (Sorted token & Compound number-name)
        recovery_hits = defaultdict(int)
        if mode in ["adaptive_b", "adaptive_c", "adaptive_d"]:
            sort_k = get_sorted_token_key(name_tokens)
            if sort_k and sort_k in self.index_sorted_token:
                for tidx in self.index_sorted_token[sort_k]:
                    recovery_hits[tidx] += 6
            
            comp_keys = get_compound_num_token_keys(name_tokens, addr_numbers)
            for ck in comp_keys:
                if ck in self.index_compound_num_name:
                    for tidx in self.index_compound_num_name[ck]:
                        recovery_hits[tidx] += 8

        # --- DYNAMIC QUOTA CALCULATION ---
        has_addr = bool(addr_norm and addr_numbers)
        name_collisions = len(name_hits)

        if mode == "baseline":
            k_name, k_addr, k_char, k_rec = 25, 15, 10, 0
        elif mode == "adaptive_a": # Fixed ceiling 50, adaptive distribution across existing 3 views
            if not has_addr or len(addr_norm) < 10:
                k_name, k_addr, k_char, k_rec = 35, 5, 10, 0
            elif name_collisions > 25:
                k_name, k_addr, k_char, k_rec = 30, 15, 5, 0
            else:
                k_name, k_addr, k_char, k_rec = 25, 15, 10, 0
        elif mode == "adaptive_b": # Fixed ceiling 50, with recovery view
            if not has_addr or len(addr_norm) < 10:
                k_name, k_addr, k_char, k_rec = 30, 5, 7, 8
            elif name_collisions > 25:
                k_name, k_addr, k_char, k_rec = 24, 14, 4, 8
            else:
                k_name, k_addr, k_char, k_rec = 22, 14, 6, 8
        elif mode == "adaptive_c": # Budget ceiling 60
            if not has_addr or len(addr_norm) < 10:
                k_name, k_addr, k_char, k_rec = 38, 5, 7, 10
            elif name_collisions > 25:
                k_name, k_addr, k_char, k_rec = 30, 15, 5, 10
            else:
                k_name, k_addr, k_char, k_rec = 26, 16, 8, 10
        elif mode == "adaptive_d": # Budget ceiling 75
            if not has_addr or len(addr_norm) < 10:
                k_name, k_addr, k_char, k_rec = 48, 7, 8, 12
            elif name_collisions > 25:
                k_name, k_addr, k_char, k_rec = 38, 18, 7, 12
            else:
                k_name, k_addr, k_char, k_rec = 32, 20, 11, 12
        else:
            k_name, k_addr, k_char, k_rec = 25, 15, 10, 0

        # Form independent candidate sets
        top_name = {tidx for tidx, _ in sorted(name_hits.items(), key=lambda x: x[1], reverse=True)[:k_name]}
        top_addr = {tidx for tidx, _ in sorted(addr_hits.items(), key=lambda x: x[1], reverse=True)[:k_addr]}
        top_char = {tidx for tidx, _ in sorted(char_hits.items(), key=lambda x: x[1], reverse=True)[:k_char]}
        top_rec  = {tidx for tidx, _ in sorted(recovery_hits.items(), key=lambda x: x[1], reverse=True)[:k_rec]} if k_rec > 0 else set()

        # Pure mathematical union
        union_tindices = top_name | top_addr | top_char | top_rec
        if not union_tindices:
            return []

        results = []
        for tidx in union_tindices:
            cid = self.targets[tidx][0]
            views_hit = 0
            if tidx in top_name: views_hit += 1
            if tidx in top_addr: views_hit += 1
            if tidx in top_char: views_hit += 1
            if tidx in top_rec:  views_hit += 1
            results.append((cid, views_hit))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
