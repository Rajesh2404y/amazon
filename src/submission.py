import os
import logging
from typing import Dict, List, Set, Iterable, Tuple

logger = logging.getLogger("entity_resolution")

def write_tsv_submission(
    matching_dict: Dict[str, List[str]],
    candidate_dict: Dict[str, List[str]],
    ordered_s1_ids: Iterable[str],
    output_dir: str = "output"
) -> Tuple[str, str]:
    """
    Writes matching_results.tsv and candidate_pairs.tsv strictly adhering to challenge specifications.
    Ensures:
      - UTF-8 encoding
      - Single tab delimiter
      - No duplicate IDs
      - Singletons have empty string after tab
      - Every S1 appears exactly once in the specified order
      - Final matches are a strict subset of candidate pairs
    """
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    logger.info(f"Writing submission files to {output_dir}...")
    
    with open(matching_path, "w", encoding="utf-8", newline="\n") as f_match, \
         open(candidate_path, "w", encoding="utf-8", newline="\n") as f_cand:
        
        # Headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        row_count = 0
        match_count = 0
        singleton_count = 0

        for s1_id in ordered_s1_ids:
            # Candidates: deduplicate preserving order
            raw_cands = candidate_dict.get(s1_id, [])
            seen_c = set()
            clean_cands = []
            for cid in raw_cands:
                if cid not in seen_c and (cid.startswith("S2-") or cid.startswith("S3-")):
                    seen_c.add(cid)
                    clean_cands.append(cid)

            # Matches: deduplicate and ensure they exist in candidates
            raw_matches = matching_dict.get(s1_id, [])
            seen_m = set()
            clean_matches = []
            for mid in raw_matches:
                if mid not in seen_m and (mid.startswith("S2-") or mid.startswith("S3-")):
                    seen_m.add(mid)
                    clean_matches.append(mid)
                    if mid not in seen_c:
                        clean_cands.append(mid)
                        seen_c.add(mid)

            cand_str = ",".join(clean_cands)
            match_str = ",".join(clean_matches)

            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{cand_str}\n")

            row_count += 1
            if match_str:
                match_count += 1
            else:
                singleton_count += 1

    logger.info(
        f"Wrote {row_count:,} rows: {match_count:,} with matches, {singleton_count:,} singletons."
    )
    return matching_path, candidate_path
