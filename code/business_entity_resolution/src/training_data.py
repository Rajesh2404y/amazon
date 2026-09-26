import os
import time
import random
import logging
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from src.normalization import normalize_business_name, normalize_business_address
from src.blocking import MultiViewBlockingIndex, extract_addr_blocking_keys, RE_DBA
from src.features import extract_pair_features, FEATURE_NAMES
from src.data_loader import stream_tsv_records, load_ground_truth
from src.memory_utils import force_garbage_collection, log_memory, get_memory_usage_mb, get_system_available_ram_gb

logger = logging.getLogger("entity_resolution")

def sample_source1_entities(
    s1_path: str,
    target_country: str,
    sample_size: int
) -> List[Dict[str, Any]]:
    """Loads a sample of Source 1 entities for the target country."""
    logger.info(f"[1/7] Loading {sample_size:,} Source 1 entities for country: {target_country}")
    s1_sample = []
    for rec in stream_tsv_records(s1_path, target_country=target_country, max_records=sample_size):
        s1_sample.append(rec)
    logger.info(f"Loaded {len(s1_sample):,} S1 entities. (RAM: {get_memory_usage_mb():.1f} MB)")
    return s1_sample

def build_training_target_pool(
    train_dir: str,
    target_country: str,
    needed_gt_ids: Set[str],
    s1_pfxs: Set[str],
    s1_two_tokens: Set[str],
    s1_addr_keys: Set[str]
) -> Tuple[MultiViewBlockingIndex, int, int]:
    """
    Builds a multi-view blocking index containing all ground-truth matches for the sample,
    plus hard-negative candidates sharing prefixes, tokens, DBA, or address keys.
    """
    t0 = time.time()
    logger.info(f"[3/7] Building multi-view blocking index for {target_country}...")
    index = MultiViewBlockingIndex(country=target_country)

    total_s2_scanned = 0
    total_s3_scanned = 0

    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(train_dir, src_file)
        logger.info(f"  [Blocking] Scanning {src_file}...")
        
        scanned = 0
        indexed = 0
        
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            col_id = header.index("entity_id")
            col_name = header.index("business_name")
            col_addr = header.index("business_address")
            col_country = header.index("country")
            
            for line in f:
                scanned += 1
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(col_id, col_name, col_country):
                    continue
                if parts[col_country] != target_country:
                    continue
                    
                eid = parts[col_id]
                bname = parts[col_name]
                baddr = parts[col_addr] if len(parts) > col_addr else ""
                
                is_gt = eid in needed_gt_ids
                is_cand = False
                
                if not is_gt:
                    n_clean = bname.strip().lower()
                    # 1. DBA Name check
                    dba_m = RE_DBA.search(n_clean)
                    if dba_m:
                        dba_clean = dba_m.group(1).replace(" ", "").replace("&", "")
                        if len(dba_clean) >= 7 and dba_clean[:7] in s1_pfxs:
                            is_cand = True
                            
                    # 2. Compact 7-char prefix
                    if not is_cand and len(n_clean) >= 7:
                        clean_comp = n_clean.replace(" ", "").replace("&", "")
                        if len(clean_comp) >= 7 and clean_comp[:7] in s1_pfxs:
                            is_cand = True

                    # 3. Two-token prefix & Token Swap
                    if not is_cand and s1_two_tokens:
                        n_words = n_clean.split()
                        if len(n_words) >= 2:
                            if f"{n_words[0]}_{n_words[1]}" in s1_two_tokens or f"{n_words[1]}_{n_words[0]}" in s1_two_tokens:
                                is_cand = True
                        elif len(n_words) == 1 and n_words[0] in s1_two_tokens:
                            is_cand = True

                    # 4. Address keys (anywhere in address)
                    if not is_cand and baddr and s1_addr_keys:
                        cand_akeys = extract_addr_blocking_keys(baddr)
                        if cand_akeys & s1_addr_keys:
                            is_cand = True

                # Add to index if GT match or plausible hard negative candidate
                if is_gt or is_cand:
                    n_norm, n_comp, n_tok, _ = normalize_business_name(bname)
                    a_norm, _, a_tok, a_num = normalize_business_address(baddr)
                    index.add_target_record(
                        eid, n_norm, n_comp, n_tok, a_norm, a_tok, a_num,
                        raw_name=bname, raw_addr=baddr
                    )
                    indexed += 1
                    
                if scanned % 1000000 == 0:
                    logger.info(f"    Scanned {scanned:,} rows -> {indexed:,} relevant records indexed (RAM: {get_memory_usage_mb():.1f} MB)")

        if src_file == "train_source2.tsv":
            total_s2_scanned = scanned
        else:
            total_s3_scanned = scanned

        logger.info(f"  Finished {src_file}: scanned {scanned:,} rows, indexed {indexed:,} target records.")

    elapsed = time.time() - t0
    logger.info(f"[TIMER] Blocking index built in {elapsed:.2f}s: {len(index.targets):,} records (RAM: {get_memory_usage_mb():.1f} MB)")
    return index, total_s2_scanned, total_s3_scanned

def generate_training_dataset(
    train_dir: str,
    target_country: str = "US",
    sample_s1_count: int = 5000,
    neg_to_pos_ratio: float = 3.0,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Fast, memory-efficient training data generator with progress logging and timers.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    t_start = time.time()
    
    # 1. Load S1 Sample
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s1_sample = sample_source1_entities(s1_path, target_country, sample_s1_count)
    sample_s1_ids = {rec["entity_id"] for rec in s1_sample}

    # 2. Load Ground Truth for Sample
    logger.info("[2/7] Loading ground truth for sampled entities...")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    gt_map = load_ground_truth(gt_path, filter_s1_ids=sample_s1_ids)

    needed_gt_ids = set()
    total_gt_matches_in_sample = 0
    for s1_id in sample_s1_ids:
        matches = gt_map.get(s1_id, [])
        total_gt_matches_in_sample += len(matches)
        for m in matches:
            needed_gt_ids.add(m)
    logger.info(f"Sample has {len(sample_s1_ids):,} S1 entities with {total_gt_matches_in_sample:,} ground truth matches ({len(needed_gt_ids):,} unique target IDs).")

    # Collect multi-view blocking keys from sample
    s1_pfxs = {s["name_compact"][:7] for s in s1_sample if len(s["name_compact"]) >= 7}
    
    s1_two_tokens = set()
    for s in s1_sample:
        toks = s["name_tokens"]
        if len(toks) >= 2:
            s1_two_tokens.add(f"{toks[0]}_{toks[1]}")
            s1_two_tokens.add(f"{toks[1]}_{toks[0]}")
        elif len(toks) == 1:
            s1_two_tokens.add(toks[0])

    s1_addr_keys = set()
    for s in s1_sample:
        s1_addr_keys.update(extract_addr_blocking_keys(s["addr_raw"]))

    # 3. Build Multi-View Blocking Index
    index, s2_scanned, s3_scanned = build_training_target_pool(
        train_dir=train_dir,
        target_country=target_country,
        needed_gt_ids=needed_gt_ids,
        s1_pfxs=s1_pfxs,
        s1_two_tokens=s1_two_tokens,
        s1_addr_keys=s1_addr_keys
    )

    target_lookup = {t[0]: t for t in index.targets}

    # 4. Generate Candidates & Mining Hard Negatives
    logger.info("[4/7] Generating candidates and measuring blocking recall...")
    t_cand_start = time.time()
    
    retrieved_gt_matches = 0
    total_candidates_generated = 0
    candidate_counts = []
    
    X_rows = []
    y_rows = []
    group_s1 = []

    report_interval = max(sample_s1_count // 5, 200)

    for i, s1_rec in enumerate(s1_sample, start=1):
        s1_id = s1_rec["entity_id"]
        true_matches = set(gt_map.get(s1_id, []))

        candidates = index.query_s1_entity(
            s1_rec["name_norm"],
            s1_rec["name_compact"],
            s1_rec["name_tokens"],
            s1_rec["addr_norm"],
            s1_rec["addr_tokens"],
            s1_rec["addr_numbers"],
            raw_addr=s1_rec["addr_raw"]
        )
        cand_count = len(candidates)
        total_candidates_generated += cand_count
        candidate_counts.append(cand_count)
        cand_dict = {c[0]: c[1] for c in candidates}

        # Blocking recall check
        pos_found = true_matches & set(cand_dict.keys())
        retrieved_gt_matches += len(pos_found)

        # Positives
        for pos_id in pos_found:
            t_rec = target_lookup.get(pos_id)
            if not t_rec:
                continue
            feats = extract_pair_features(
                s1_rec["name_norm"], s1_rec["name_compact"], s1_rec["name_tokens"],
                s1_rec["addr_norm"], s1_rec["addr_tokens"], s1_rec["addr_numbers"],
                pos_id, t_rec[1], t_rec[2], t_rec[3], t_rec[4], t_rec[5], t_rec[6],
                rule_hits=cand_dict[pos_id]
            )
            X_rows.append(feats)
            y_rows.append(1)
            group_s1.append(s1_id)

        # Hard Negatives
        hard_negs = [cid for cid in cand_dict if cid not in true_matches]
        num_negs = int(len(pos_found) * neg_to_pos_ratio)
        if num_negs == 0 and not true_matches and hard_negs:
            num_negs = 2
        sampled_negs = random.sample(hard_negs, min(len(hard_negs), num_negs))
        
        for neg_id in sampled_negs:
            t_rec = target_lookup.get(neg_id)
            if not t_rec:
                continue
            feats = extract_pair_features(
                s1_rec["name_norm"], s1_rec["name_compact"], s1_rec["name_tokens"],
                s1_rec["addr_norm"], s1_rec["addr_tokens"], s1_rec["addr_numbers"],
                neg_id, t_rec[1], t_rec[2], t_rec[3], t_rec[4], t_rec[5], t_rec[6],
                rule_hits=cand_dict[neg_id]
            )
            X_rows.append(feats)
            y_rows.append(0)
            group_s1.append(s1_id)

        if i % report_interval == 0 or i == len(s1_sample):
            avg_cands = total_candidates_generated / i
            logger.info(
                f"  [Candidate Gen] S1: {i:,} / {len(s1_sample):,} | "
                f"Candidates: {total_candidates_generated:,} (Avg: {avg_cands:.1f}/S1) | "
                f"RAM: {get_memory_usage_mb():.1f} MB"
            )

    t_cand = time.time() - t_cand_start
    blocking_recall = (retrieved_gt_matches / total_gt_matches_in_sample) if total_gt_matches_in_sample > 0 else 0.0
    cand_arr = np.array(candidate_counts) if candidate_counts else np.array([0])
    avg_cands_final = float(cand_arr.mean())
    median_cands = float(np.median(cand_arr))
    max_cands = int(cand_arr.max())
    p90_cands = float(np.percentile(cand_arr, 90))
    p95_cands = float(np.percentile(cand_arr, 95))
    p99_cands = float(np.percentile(cand_arr, 99))

    total_pool_size = s2_scanned + s3_scanned
    reduction_ratio = 1.0 - (avg_cands_final / total_pool_size) if total_pool_size > 0 else 1.0

    logger.info(f"[TIMER] Candidate & Feature generation completed in {t_cand:.2f}s")
    logger.info(
        f"Blocking Performance:\n"
        f"  Total GT Matches in Sample: {total_gt_matches_in_sample:,}\n"
        f"  Retrieved GT Matches:      {retrieved_gt_matches:,}\n"
        f"  Blocking Recall:           {blocking_recall*100:.2f}%\n"
        f"  Average Candidates per S1: {avg_cands_final:.2f}\n"
        f"  Median Candidates per S1:  {median_cands:.1f}\n"
        f"  Max Candidates per S1:     {max_cands}\n"
        f"  Candidate Reduction:       {reduction_ratio*100:.5f}%\n"
        f"  Total Training Pairs:      {len(y_rows):,} ({sum(y_rows):,} pos, {len(y_rows)-sum(y_rows):,} neg)"
    )

    stats = {
        "s1_processed": len(s1_sample),
        "s2_scanned": s2_scanned,
        "s3_scanned": s3_scanned,
        "relevant_indexed": len(index.targets),
        "total_gt_matches": total_gt_matches_in_sample,
        "retrieved_gt_matches": retrieved_gt_matches,
        "blocking_recall": round(blocking_recall, 4),
        "total_candidates": total_candidates_generated,
        "avg_candidates_per_s1": round(avg_cands_final, 2),
        "median_candidates_per_s1": round(median_cands, 2),
        "max_candidates_per_s1": max_cands,
        "p90_candidates": round(p90_cands, 2),
        "p95_candidates": round(p95_cands, 2),
        "p99_candidates": round(p99_cands, 2),
        "candidate_reduction_ratio": round(reduction_ratio, 6),
        "total_training_pairs": len(y_rows),
        "positive_pairs": sum(y_rows),
        "negative_pairs": len(y_rows) - sum(y_rows),
        "candidate_gen_time_sec": round(t_cand, 2),
        "runtime_sec": round(time.time() - t_start, 2)
    }

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int32)
    return X, y, group_s1, stats
