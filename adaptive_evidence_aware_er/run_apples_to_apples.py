import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import random
import logging
import psutil
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.config import config
from src.features import extract_pair_features
from src.model import EntityXGBoostMatcher
from src.evaluation import evaluate_macro_f05
from src.router import EntityRouter
from src.data_loader import load_ground_truth
from src.training_data import sample_source1_entities
from src.memory_utils import get_memory_usage_mb, force_garbage_collection

from src.blocking import MultiViewBlockingIndex, extract_addr_blocking_keys, RE_DBA
from src.normalization import normalize_business_name, normalize_business_address
from adaptive_evidence_aware_er.src.adaptive_blocking import (
    AdaptiveMultiViewBlockingIndex,
    get_sorted_token_key,
    get_compound_num_token_keys
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("apples_to_apples")

def run_comparison():
    train_dir = "dataset/train"
    target_country = "US"
    sample_s1_count = 15000

    logger.info("==================================================")
    logger.info("STRICT APPLES-TO-APPLES EVALUATION: BASELINE VS ADAPTIVE-C")
    logger.info("==================================================")

    # 1. Load exact 15,000 S1 sample
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s1_sample = sample_source1_entities(s1_path, target_country, sample_s1_count)
    sample_s1_ids = {rec["entity_id"] for rec in s1_sample}

    # 2. Load ground truth
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    gt_map = load_ground_truth(gt_path, filter_s1_ids=sample_s1_ids)

    needed_gt_ids = set()
    total_gt = 0
    for s1_id in sample_s1_ids:
        m = gt_map.get(s1_id, [])
        total_gt += len(m)
        for target_id in m:
            needed_gt_ids.add(target_id)

    logger.info(f"Sample: {len(sample_s1_ids):,} entities with {total_gt:,} GT matches.")

    # Collect keys for candidate filtering
    s1_pfxs = {s["name_compact"][:7] for s in s1_sample if len(s["name_compact"]) >= 7}
    s1_two_tokens = set()
    s1_sorted_tokens = set()
    for s in s1_sample:
        toks = s["name_tokens"]
        if len(toks) >= 2:
            s1_two_tokens.add(f"{toks[0]}_{toks[1]}")
            s1_two_tokens.add(f"{toks[1]}_{toks[0]}")
        elif len(toks) == 1:
            s1_two_tokens.add(toks[0])
        sk = get_sorted_token_key(toks)
        if sk: s1_sorted_tokens.add(sk)

    s1_addr_keys = set()
    for s in s1_sample:
        s1_addr_keys.update(extract_addr_blocking_keys(s["addr_raw"]))

    # Build shared target pool for both
    logger.info("Building target pool from train_source2.tsv and train_source3.tsv...")
    t0_idx = time.time()
    
    baseline_index = MultiViewBlockingIndex(country=target_country)
    adaptive_index = AdaptiveMultiViewBlockingIndex(country=target_country)

    total_s2 = 0
    total_s3 = 0

    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(train_dir, src_file)
        scanned = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            col_id = header.index("entity_id")
            col_name = header.index("business_name")
            col_addr = header.index("business_address")
            col_country = header.index("country")

            for line in f:
                scanned += 1
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(col_id, col_name, col_country) or parts[col_country] != target_country:
                    continue

                eid = parts[col_id]
                bname = parts[col_name]
                baddr = parts[col_addr] if len(parts) > col_addr else ""

                is_gt = eid in needed_gt_ids
                is_cand = False

                if not is_gt:
                    n_clean = bname.strip().lower()
                    dba_m = RE_DBA.search(n_clean)
                    if dba_m:
                        dba_clean = dba_m.group(1).replace(" ", "").replace("&", "")
                        if len(dba_clean) >= 7 and dba_clean[:7] in s1_pfxs:
                            is_cand = True

                    if not is_cand and len(n_clean) >= 7:
                        clean_comp = n_clean.replace(" ", "").replace("&", "")
                        if len(clean_comp) >= 7 and clean_comp[:7] in s1_pfxs:
                            is_cand = True

                    if not is_cand and s1_two_tokens:
                        n_words = n_clean.split()
                        if len(n_words) >= 2:
                            if f"{n_words[0]}_{n_words[1]}" in s1_two_tokens or f"{n_words[1]}_{n_words[0]}" in s1_two_tokens:
                                is_cand = True
                        elif len(n_words) == 1 and n_words[0] in s1_two_tokens:
                            is_cand = True

                    if not is_cand and baddr and s1_addr_keys:
                        if extract_addr_blocking_keys(baddr) & s1_addr_keys:
                            is_cand = True

                    if not is_cand and s1_sorted_tokens:
                        sk = get_sorted_token_key(tuple(n_clean.split()))
                        if sk and sk in s1_sorted_tokens:
                            is_cand = True

                if is_gt or is_cand:
                    n_norm, n_comp, n_tok, _ = normalize_business_name(bname)
                    a_norm, _, a_tok, a_num = normalize_business_address(baddr)
                    baseline_index.add_target_record(
                        eid, n_norm, n_comp, n_tok, a_norm, a_tok, a_num,
                        raw_name=bname, raw_addr=baddr
                    )
                    adaptive_index.add_target_record(
                        eid, n_norm, n_comp, n_tok, a_norm, a_tok, a_num,
                        raw_name=bname, raw_addr=baddr
                    )

        if src_file == "train_source2.tsv": total_s2 = scanned
        else: total_s3 = scanned

    logger.info(f"Target pool built in {time.time() - t0_idx:.2f}s: {len(baseline_index.targets):,} records.")
    total_target_pool = total_s2 + total_s3
    target_lookup = {t[0]: t for t in baseline_index.targets}

    # Helper function to run one pipeline configuration
    def evaluate_configuration(name, query_func):
        random.seed(42)
        np.random.seed(42)
        t_start = time.time()
        logger.info(f"--- Running {name} ---")

        retrieved_gt = 0
        cand_counts = []
        X_rows = []
        y_rows = []
        group_s1 = []

        for s1_rec in s1_sample:
            s1_id = s1_rec["entity_id"]
            true_matches = set(gt_map.get(s1_id, []))

            candidates = query_func(s1_rec)
            cand_cnt = len(candidates)
            cand_counts.append(cand_cnt)
            cand_dict = {c[0]: c[1] for c in candidates}

            pos_found = true_matches & set(cand_dict.keys())
            retrieved_gt += len(pos_found)

            for pos_id in pos_found:
                t_rec = target_lookup.get(pos_id)
                if not t_rec: continue
                feats = extract_pair_features(
                    s1_rec["name_norm"], s1_rec["name_compact"], s1_rec["name_tokens"],
                    s1_rec["addr_norm"], s1_rec["addr_tokens"], s1_rec["addr_numbers"],
                    pos_id, t_rec[1], t_rec[2], t_rec[3], t_rec[4], t_rec[5], t_rec[6],
                    rule_hits=cand_dict[pos_id]
                )
                X_rows.append(feats)
                y_rows.append(1)
                group_s1.append(s1_id)

            hard_negs = [cid for cid in cand_dict if cid not in true_matches]
            num_negs = int(len(pos_found) * 3.0)
            if num_negs == 0 and not true_matches and hard_negs:
                num_negs = 2
            sampled_negs = random.sample(hard_negs, min(len(hard_negs), num_negs))

            for neg_id in sampled_negs:
                t_rec = target_lookup.get(neg_id)
                if not t_rec: continue
                feats = extract_pair_features(
                    s1_rec["name_norm"], s1_rec["name_compact"], s1_rec["name_tokens"],
                    s1_rec["addr_norm"], s1_rec["addr_tokens"], s1_rec["addr_numbers"],
                    neg_id, t_rec[1], t_rec[2], t_rec[3], t_rec[4], t_rec[5], t_rec[6],
                    rule_hits=cand_dict[neg_id]
                )
                X_rows.append(feats)
                y_rows.append(0)
                group_s1.append(s1_id)

        blocking_recall = retrieved_gt / total_gt
        c_arr = np.array(cand_counts)
        avg_cands = float(c_arr.mean())
        med_cands = float(np.median(c_arr))
        max_cands = int(c_arr.max())

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.int32)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=group_s1))

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        val_groups = [group_s1[idx] for idx in val_idx]

        matcher = EntityXGBoostMatcher(config.xgboost_cfg)
        matcher.fit(X_train, y_train, X_val, y_val)
        val_probs = matcher.predict_proba(X_val)

        val_gt = {}
        for idx, s1_id in enumerate(val_groups):
            if s1_id not in val_gt: val_gt[s1_id] = set()
            if y_val[idx] == 1: val_gt[s1_id].add(f"CAND_{idx}")

        val_entity_scores = {}
        for idx, s1_id in enumerate(val_groups):
            if s1_id not in val_entity_scores: val_entity_scores[s1_id] = []
            val_entity_scores[s1_id].append((f"CAND_{idx}", float(val_probs[idx])))

        # Exact Router optimization as in baseline train.py
        router = EntityRouter()
        opt = router.optimize_threshold(val_entity_scores, val_gt, threshold_range=(0.10, 0.95, 0.02))

        elapsed = time.time() - t_start
        peak_ram = get_memory_usage_mb()

        del X, y, X_train, y_train, X_val, y_val, matcher
        force_garbage_collection()

        return {
            "name": name,
            "blocking_recall": blocking_recall,
            "retrieved_gt": retrieved_gt,
            "total_gt": total_gt,
            "avg_candidates": avg_cands,
            "median_candidates": med_cands,
            "max_candidates": max_cands,
            "total_pairs": len(y_rows),
            "precision": opt["macro_precision"],
            "recall": opt["macro_recall"],
            "macro_f05": opt["macro_f05"],
            "optimal_threshold": opt["optimal_threshold"],
            "singleton_accuracy": opt["singleton_accuracy"],
            "runtime_sec": elapsed,
            "peak_ram_mb": peak_ram
        }

    # 1. Run Frozen Baseline
    res_baseline = evaluate_configuration(
        "Frozen Baseline (Fixed 25/15/10, Ceiling 50)",
        lambda s: baseline_index.query_s1_entity(
            s["name_norm"], s["name_compact"], s["name_tokens"],
            s["addr_norm"], s["addr_tokens"], s["addr_numbers"],
            raw_addr=s["addr_raw"]
        )
    )

    # 2. Run Adaptive-C
    res_adaptive_c = evaluate_configuration(
        "Adaptive-C (Dynamic Quotas + Recovery Views, Ceiling 60)",
        lambda s: adaptive_index.query_s1_adaptive(
            s["name_norm"], s["name_compact"], s["name_tokens"],
            s["addr_norm"], s["addr_tokens"], s["addr_numbers"],
            raw_addr=s["addr_raw"],
            mode="adaptive_c",
            max_budget=60
        )
    )

    delta_f05 = res_adaptive_c["macro_f05"] - res_baseline["macro_f05"]
    delta_recall = res_adaptive_c["blocking_recall"] - res_baseline["blocking_recall"]

    logger.info("==================================================")
    logger.info(f"FINAL COMPARISON:")
    logger.info(f"  Frozen Baseline Macro F0.5 : {res_baseline['macro_f05']:.4f} (Threshold: {res_baseline['optimal_threshold']:.2f})")
    logger.info(f"  Adaptive-C Macro F0.5      : {res_adaptive_c['macro_f05']:.4f} (Threshold: {res_adaptive_c['optimal_threshold']:.2f})")
    logger.info(f"  Delta Macro F0.5           : {delta_f05:+.4f}")
    logger.info(f"  Blocking Recall Delta      : {delta_recall*100:+.2f}%")
    logger.info("==================================================")

    # Write reports/adaptive_c_vs_frozen_baseline.md
    with open("reports/adaptive_c_vs_frozen_baseline.md", "w", encoding="utf-8") as f:
        f.write("# Strict Apples-to-Apples Comparison: Frozen Baseline vs. Adaptive-C\n\n")
        f.write("## 1. Experimental Methodology & Invariants\n\n")
        f.write("Both pipelines were evaluated on the identical 15,000 Source 1 validation sample using:\n")
        f.write("- Identical random seed (`42`)\n")
        f.write("- Identical 75%/25% `GroupShuffleSplit` partitioned by Source 1 ID\n")
        f.write("- Identical 1:3 positive-to-hard-negative sampling policy\n")
        f.write("- Identical 19-dimensional RapidFuzz feature extraction kernel\n")
        f.write("- Identical XGBoost HistClassifier configuration\n")
        f.write("- Identical threshold calibration search range `(0.10, 0.95, 0.02)` maximizing Macro F0.5\n")
        f.write("- Identical competition Macro F0.5 evaluation function\n\n")

        f.write("## 2. Quantitative Comparison Table\n\n")
        f.write("| Metric | Frozen Baseline | Adaptive-C | Delta |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Blocking Recall** | {res_baseline['blocking_recall']*100:.2f}% ({res_baseline['retrieved_gt']:,} / {res_baseline['total_gt']:,}) | {res_adaptive_c['blocking_recall']*100:.2f}% ({res_adaptive_c['retrieved_gt']:,} / {res_adaptive_c['total_gt']:,}) | **{delta_recall*100:+.2f}%** ({res_adaptive_c['retrieved_gt'] - res_baseline['retrieved_gt']:+,} matches) |\n")
        f.write(f"| **Average Candidates / S1** | {res_baseline['avg_candidates']:.2f} | {res_adaptive_c['avg_candidates']:.2f} | {res_adaptive_c['avg_candidates'] - res_baseline['avg_candidates']:+.2f} |\n")
        f.write(f"| **Median Candidates / S1** | {res_baseline['median_candidates']:.1f} | {res_adaptive_c['median_candidates']:.1f} | {res_adaptive_c['median_candidates'] - res_baseline['median_candidates']:+.1f} |\n")
        f.write(f"| **Maximum Candidates / S1** | {res_baseline['max_candidates']} | {res_adaptive_c['max_candidates']} | {res_adaptive_c['max_candidates'] - res_baseline['max_candidates']:+d} |\n")
        f.write(f"| **Total Training Pairs** | {res_baseline['total_pairs']:,} | {res_adaptive_c['total_pairs']:,} | {res_adaptive_c['total_pairs'] - res_baseline['total_pairs']:+,} |\n")
        f.write(f"| **Validation Precision** | {res_baseline['precision']:.4f} | {res_adaptive_c['precision']:.4f} | {res_adaptive_c['precision'] - res_baseline['precision']:+.4f} |\n")
        f.write(f"| **Validation Recall** | {res_baseline['recall']:.4f} | {res_adaptive_c['recall']:.4f} | {res_adaptive_c['recall'] - res_baseline['recall']:+.4f} |\n")
        f.write(f"| **Validation Macro F0.5** | **{res_baseline['macro_f05']:.4f}** | **{res_adaptive_c['macro_f05']:.4f}** | **{delta_f05:+.4f}** |\n")
        f.write(f"| **Optimal Decision Threshold** | {res_baseline['optimal_threshold']:.2f} | {res_adaptive_c['optimal_threshold']:.2f} | {res_adaptive_c['optimal_threshold'] - res_baseline['optimal_threshold']:+.2f} |\n")
        f.write(f"| **Singleton Accuracy** | {res_baseline['singleton_accuracy']*100:.2f}% | {res_adaptive_c['singleton_accuracy']*100:.2f}% | {(res_adaptive_c['singleton_accuracy'] - res_baseline['singleton_accuracy'])*100:+.2f}% |\n")
        f.write(f"| **Pipeline Runtime** | {res_baseline['runtime_sec']:.1f}s | {res_adaptive_c['runtime_sec']:.1f}s | {res_adaptive_c['runtime_sec'] - res_baseline['runtime_sec']:+.1f}s |\n")
        f.write(f"| **Peak Process RAM** | {res_baseline['peak_ram_mb']:.1f} MB | {res_adaptive_c['peak_ram_mb']:.1f} MB | {res_adaptive_c['peak_ram_mb'] - res_baseline['peak_ram_mb']:+.1f} MB |\n\n")

        f.write("## 3. Formal Acceptance Determination\n\n")
        if res_adaptive_c['macro_f05'] > res_baseline['macro_f05'] and res_adaptive_c['macro_f05'] >= 0.9645:
            f.write("### Decision: **ACCEPT**\n")
            f.write(f"Adaptive-C achieves Macro F0.5 = **{res_adaptive_c['macro_f05']:.4f}**, exceeding the frozen baseline ({res_baseline['macro_f05']:.4f}) by **{delta_f05:+.4f}** ")
            f.write(f"while lifting blocking recall from {res_baseline['blocking_recall']*100:.2f}% to {res_adaptive_c['blocking_recall']*100:.2f}% (+{delta_recall*100:.2f}%) ")
            f.write(f"with high precision ({res_adaptive_c['precision']:.4f}) and controlled candidate density (Avg: {res_adaptive_c['avg_candidates']:.2f}).\n")
        else:
            f.write("### Decision: **REJECT**\n")
            f.write(f"Adaptive-C does not beat the frozen 0.9645 baseline under identical conditions.\n")

    logger.info("Saved reports/adaptive_c_vs_frozen_baseline.md")

if __name__ == "__main__":
    run_comparison()
