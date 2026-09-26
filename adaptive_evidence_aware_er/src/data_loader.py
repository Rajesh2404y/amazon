import os
import sys
import logging
from typing import Generator, Dict, List, Set, Optional, Tuple, Any
from src.normalization import normalize_business_name, normalize_business_address
from src.memory_utils import force_garbage_collection, log_memory

logger = logging.getLogger("entity_resolution")

def stream_tsv_records(
    file_path: str,
    target_country: Optional[str] = None,
    max_records: Optional[int] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Streams records from a TSV file one row at a time.
    Yields dictionary with raw and normalized fields.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    count = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        col_id = header.index("entity_id")
        col_name = header.index("business_name")
        col_addr = header.index("business_address")
        col_country = header.index("country")
        
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= max(col_id, col_name, col_country):
                continue
                
            country = parts[col_country]
            if target_country and country != target_country:
                continue
                
            eid = parts[col_id]
            bname = parts[col_name]
            baddr = parts[col_addr] if len(parts) > col_addr else ""
            
            # Compute normalized views
            name_norm, name_compact, name_tokens, name_ngrams = normalize_business_name(bname)
            addr_norm, addr_compact, addr_tokens, addr_numbers = normalize_business_address(baddr)
            
            yield {
                "entity_id": eid,
                "country": country,
                "name_raw": bname,
                "name_norm": name_norm,
                "name_compact": name_compact,
                "name_tokens": name_tokens,
                "name_ngrams": name_ngrams,
                "addr_raw": baddr,
                "addr_norm": addr_norm,
                "addr_compact": addr_compact,
                "addr_tokens": addr_tokens,
                "addr_numbers": addr_numbers
            }
            
            count += 1
            if max_records and count >= max_records:
                break

def load_ground_truth(gt_path: str, filter_s1_ids: Optional[Set[str]] = None) -> Dict[str, List[str]]:
    """Loads ground truth mapping s1_id -> list of matched IDs."""
    gt_map = {}
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline() # header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if filter_s1_ids is not None and s1_id not in filter_s1_ids:
                continue
            matched_str = parts[1].strip() if len(parts) > 1 else ""
            gt_map[s1_id] = [m.strip() for m in matched_str.split(",") if m.strip()]
    return gt_map
