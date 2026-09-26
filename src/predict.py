import os
import json
import logging
from typing import Dict, List, Set, Optional
from src.config import config
from src.normalization import normalize_business_name, normalize_business_address
from src.blocking import MultiViewBlockingIndex
from src.features import extract_pair_features
from src.model import EntityXGBoostMatcher
from src.router import EntityRouter
from src.submission import write_tsv_submission
from src.data_loader import stream_tsv_records
from src.memory_utils import log_memory, force_garbage_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("entity_resolution")

def build_test_country_index(
    test_dir: str,
    country: str,
    max_records: Optional[int] = None
) -> MultiViewBlockingIndex:
    """
    Builds blocking index for test S2 and S3 for a specific country.
    """
    logger.info(f"Building Test Blocking Index for country: {country}")
    index = MultiViewBlockingIndex(country=country)

    s2_path = os.path.join(test_dir, "test_source2.tsv")
    count_s2 = 0
    for rec in stream_tsv_records(s2_path, target_country=country, max_records=max_records):
        index.add_target_record(
            rec["entity_id"],
            rec["name_norm"],
            rec["name_compact"],
            rec["name_tokens"],
            rec["addr_norm"],
            rec["addr_tokens"],
            rec["addr_numbers"],
            raw_name=rec["name_raw"],
            raw_addr=rec["addr_raw"]
        )
        count_s2 += 1

    s3_path = os.path.join(test_dir, "test_source3.tsv")
    count_s3 = 0
    for rec in stream_tsv_records(s3_path, target_country=country, max_records=max_records):
        index.add_target_record(
            rec["entity_id"],
            rec["name_norm"],
            rec["name_compact"],
            rec["name_tokens"],
            rec["addr_norm"],
            rec["addr_tokens"],
            rec["addr_numbers"],
            raw_name=rec["name_raw"],
            raw_addr=rec["addr_raw"]
        )
        count_s3 += 1

    logger.info(f"Test index for {country}: {count_s2:,} S2 + {count_s3:,} S3 = {len(index.targets):,} total records.")
    return index

def run_test_inference(
    test_dir: str = "dataset/test",
    model_path: str = "models/xgboost_matcher.joblib",
    metadata_path: str = "models/model_metadata.json",
    output_dir: str = "output",
    max_s1_per_country: Optional[int] = None,
    max_targets_per_country: Optional[int] = None
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load trained model & threshold
    logger.info(f"Loading trained matcher from {model_path}...")
    matcher = EntityXGBoostMatcher()
    matcher.load(model_path)

    threshold = 0.58
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            threshold = meta.get("optimal_threshold", 0.58)
    logger.info(f"Using calibrated decision threshold: {threshold:.3f}")

    router = EntityRouter(score_threshold=threshold, score_margin=0.15, singleton_cutoff=0.48)

    # 2. Collect all test S1 IDs in original file order
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    ordered_s1_ids = []
    logger.info("Reading test Source 1 entity IDs...")
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.split("\t", 1)
            if parts[0].strip():
                ordered_s1_ids.append(parts[0].strip())
    logger.info(f"Total test S1 entities to process: {len(ordered_s1_ids):,}")

    final_matches: Dict[str, List[str]] = {}
    final_candidates: Dict[str, List[str]] = {}

    # 3. Process Country by Country (India, US, France)
    countries = ["India", "US", "France"]

    for country in countries:
        log_memory(f"Start processing test country: {country}")
        index = build_test_country_index(test_dir, country=country, max_records=max_targets_per_country)
        target_lookup = {t[0]: t for t in index.targets}

        # Stream S1 for this country
        count_country_s1 = 0
        for s1_rec in stream_tsv_records(s1_path, target_country=country, max_records=max_s1_per_country):
            s1_id = s1_rec["entity_id"]
            candidates = index.query_s1_entity(
                s1_rec["name_norm"],
                s1_rec["name_compact"],
                s1_rec["name_tokens"],
                s1_rec["addr_norm"],
                s1_rec["addr_tokens"],
                s1_rec["addr_numbers"],
                raw_addr=s1_rec["addr_raw"]
            )
            cand_ids = [c[0] for c in candidates]
            final_candidates[s1_id] = cand_ids

            if not candidates:
                final_matches[s1_id] = []
                count_country_s1 += 1
                continue

            # Batch feature extraction for candidates of this S1
            feats_list = []
            valid_cand_ids = []
            for cid, rule_hits in candidates:
                t_rec = target_lookup.get(cid)
                if not t_rec:
                    continue
                f = extract_pair_features(
                    s1_rec["name_norm"], s1_rec["name_compact"], s1_rec["name_tokens"],
                    s1_rec["addr_norm"], s1_rec["addr_tokens"], s1_rec["addr_numbers"],
                    cid, t_rec[1], t_rec[2], t_rec[3], t_rec[4], t_rec[5], t_rec[6],
                    rule_hits=rule_hits
                )
                feats_list.append(f)
                valid_cand_ids.append(cid)

            if feats_list:
                import numpy as np
                X_pairs = np.array(feats_list, dtype=np.float32)
                probs = matcher.predict_proba(X_pairs)
                scored = list(zip(valid_cand_ids, probs.tolist()))
                matches = router.decide_matches(scored)
                final_matches[s1_id] = matches
            else:
                final_matches[s1_id] = []

            count_country_s1 += 1
            if count_country_s1 % 50000 == 0:
                log_memory(f"Processed {count_country_s1:,} {country} S1 entities")

        logger.info(f"Finished country {country}: processed {count_country_s1:,} entities.")
        del index
        del target_lookup
        force_garbage_collection()

    # 4. Generate TSV files
    m_path, c_path = write_tsv_submission(
        matching_dict=final_matches,
        candidate_dict=final_candidates,
        ordered_s1_ids=ordered_s1_ids,
        output_dir=output_dir
    )
    logger.info(f"Generated submission files:\n  Matching: {m_path}\n  Candidates: {c_path}")
    return m_path, c_path

if __name__ == "__main__":
    run_test_inference()
