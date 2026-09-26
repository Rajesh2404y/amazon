import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import csv
import random
import logging
import psutil
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.config import config
from src.normalization import normalize_business_name, normalize_business_address
from src.features import extract_pair_features
from src.model import EntityXGBoostMatcher
from src.evaluation import evaluate_macro_f05
from src.router import EntityRouter
from src.data_loader import stream_tsv_records, load_ground_truth
from src.training_data import sample_source1_entities
from src.memory_utils import get_memory_usage_mb, force_garbage_collection

from adaptive_evidence_aware_er.src.adaptive_blocking import (
    AdaptiveMultiViewBlockingIndex,
    extract_addr_blocking_keys,
    RE_DBA,
    get_sorted_token_key,
    get_compound_num_token_keys
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("adaptive_blocking")

def run_all_ablations():
    random.seed(42)
    np.random.seed(42)

    train_dir = "dataset/train"
    target_country = "US"
    sample_s1_count = 15000

    logger.info("==================================================")
    logger.info(f"STARTING 15K ADAPTIVE BLOCKING ABLATION SUITE")
    logger.info("==================================================")

    # 1. Load S1 Sample
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s1_sample = sample_source1_entities(s1_path, target_country, sample_s1_count)
    sample_s1_ids = {rec["entity_id"] for rec in s1_sample}

    # 2. Load Ground Truth
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    gt_map = load_ground_truth(gt_path, filter_s1_ids=sample_s1_ids)

    needed_gt_ids = set()
    total_gt = 0
    for s1_id in sample_s1_ids:
        matches = gt_map.get(s1_id, [])
        total_gt += len(matches)
        for m in matches:
            needed_gt_ids.add(m)

    logger.info(f"Sample has {len(sample_s1_ids):,} S1 entities with {total_gt:,} GT matches ({len(needed_gt_ids):,} unique targets).")

    # 3. Collect S1 Blocking Keys
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

    # 4. Build Adaptive Blocking Index ONCE
    logger.info("Building Adaptive Multi-View Target Index (scanned across S2 + S3)...")
    t_idx_start = time.time()
    index = AdaptiveMultiViewBlockingIndex(country=target_country)

    total_s2 = 0
    total_s3 = 0

    for src_file in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(train_dir, src_file)
        logger.info(f"  Scanning {src_file}...")
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
                if len(parts) <= max(col_id, col_name, col_country) or parts[col_country] != target_country:
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

                    # 2. 7-char prefix
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

                    # 4. Address keys
                    if not is_cand and baddr and s1_addr_keys:
                        if extract_addr_blocking_keys(baddr) & s1_addr_keys:
                            is_cand = True

                    # 5. Sorted token key (Recovery)
                    if not is_cand and s1_sorted_tokens:
                        sk = get_sorted_token_key(tuple(n_clean.split()))
                        if sk and sk in s1_sorted_tokens:
                            is_cand = True

                if is_gt or is_cand:
                    n_norm, n_comp, n_tok, _ = normalize_business_name(bname)
                    a_norm, _, a_tok, a_num = normalize_business_address(baddr)
                    index.add_target_record(
                        eid, n_norm, n_comp, n_tok, a_norm, a_tok, a_num,
                        raw_name=bname, raw_addr=baddr
                    )
                    indexed += 1

                if scanned % 1000000 == 0:
                    logger.info(f"    Scanned {scanned:,} -> indexed {indexed:,} (RAM: {get_memory_usage_mb():.1f} MB)")

        if src_file == "train_source2.tsv":
            total_s2 = scanned
        else:
            total_s3 = scanned

    t_idx = time.time() - t_idx_start
    total_target_pool = total_s2 + total_s3
    logger.info(f"Index built in {t_idx:.2f}s: {len(index.targets):,} records indexed (RAM: {get_memory_usage_mb():.1f} MB)")

    target_lookup = {t[0]: t for t in index.targets}

    # Experiments Definition
    experiments = [
        ("Baseline", "baseline", 50),
        ("Adaptive-A", "adaptive_a", 50),
        ("Adaptive-B", "adaptive_b", 50),
        ("Adaptive-C", "adaptive_c", 60),
        ("Adaptive-D", "adaptive_d", 75),
    ]

    ablation_results = []

    for exp_name, mode, budget in experiments:
        logger.info("--------------------------------------------------")
        logger.info(f"EVALUATING EXPERIMENT: {exp_name} (Mode: {mode}, Budget: {budget})")
        logger.info("--------------------------------------------------")
        t_exp_start = time.time()

        retrieved_gt = 0
        total_candidates = 0
        cand_counts = []

        X_rows = []
        y_rows = []
        group_s1 = []

        for i, s1_rec in enumerate(s1_sample, start=1):
            s1_id = s1_rec["entity_id"]
            true_matches = set(gt_map.get(s1_id, []))

            candidates = index.query_s1_adaptive(
                s1_rec["name_norm"],
                s1_rec["name_compact"],
                s1_rec["name_tokens"],
                s1_rec["addr_norm"],
                s1_rec["addr_tokens"],
                s1_rec["addr_numbers"],
                raw_addr=s1_rec["addr_raw"],
                mode=mode,
                max_budget=budget
            )

            cand_cnt = len(candidates)
            total_candidates += cand_cnt
            cand_counts.append(cand_cnt)
            cand_dict = {c[0]: c[1] for c in candidates}

            # Check blocking recall
            pos_found = true_matches & set(cand_dict.keys())
            retrieved_gt += len(pos_found)

            # Positives
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

            # Hard Negatives (1:3 ratio)
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

        # Blocking Metrics
        blocking_recall = retrieved_gt / total_gt
        c_arr = np.array(cand_counts)
        avg_cands = float(c_arr.mean())
        med_cands = float(np.median(c_arr))
        max_cands = int(c_arr.max())
        reduction_ratio = 1.0 - (avg_cands / total_target_pool)

        logger.info(f"  Blocking Recall: {blocking_recall*100:.2f}% ({retrieved_gt:,} / {total_gt:,})")
        logger.info(f"  Candidates/S1: Avg {avg_cands:.2f} | Med {med_cands:.1f} | Max {max_cands}")
        logger.info(f"  Total Pairs: {len(y_rows):,} ({sum(y_rows):,} pos, {len(y_rows)-sum(y_rows):,} neg)")

        # Downstream Model Training & Evaluation
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

        # Build validation ground truth
        val_gt = {}
        for idx, s1_id in enumerate(val_groups):
            if s1_id not in val_gt: val_gt[s1_id] = set()
            if y_val[idx] == 1: val_gt[s1_id].add(f"CAND_{idx}")

        val_entity_scores = {}
        for idx, s1_id in enumerate(val_groups):
            if s1_id not in val_entity_scores: val_entity_scores[s1_id] = []
            val_entity_scores[s1_id].append((f"CAND_{idx}", float(val_probs[idx])))

        # Threshold optimization for Macro F0.5
        best_f05 = -1.0
        best_prec = 0.0
        best_rec = 0.0
        best_thresh = 0.54
        best_sing_acc = 1.0

        for thresh in np.arange(0.10, 0.96, 0.02):
            router = EntityRouter(score_threshold=thresh, score_margin=0.15, singleton_cutoff=0.48)
            preds = {}
            for s1_id, scored in val_entity_scores.items():
                preds[s1_id] = set(router.decide_matches(scored))

            metrics = evaluate_macro_f05(preds, val_gt)
            if metrics["macro_f05"] > best_f05:
                best_f05 = metrics["macro_f05"]
                best_prec = metrics["macro_precision"]
                best_rec = metrics["macro_recall"]
                best_thresh = round(float(thresh), 2)
                best_sing_acc = metrics["singleton_accuracy"]

        t_exp = time.time() - t_exp_start
        peak_ram = get_memory_usage_mb()

        logger.info(f"  Optimal Threshold: {best_thresh:.2f}")
        logger.info(f"  Macro F0.5: {best_f05:.4f} | Prec: {best_prec:.4f} | Rec: {best_rec:.4f} | SingAcc: {best_sing_acc*100:.2f}%")
        logger.info(f"  Runtime: {t_exp:.2f}s | Process RAM: {peak_ram:.1f} MB")

        res_dict = {
            "experiment": exp_name,
            "blocking_recall": round(blocking_recall, 4),
            "avg_candidates": round(avg_cands, 2),
            "median_candidates": round(med_cands, 1),
            "max_candidates": max_cands,
            "total_pairs": len(y_rows),
            "precision": round(best_prec, 4),
            "recall": round(best_rec, 4),
            "macro_f0_5": round(best_f05, 4),
            "singleton_accuracy": round(best_sing_acc, 4),
            "runtime_seconds": round(t_exp, 2),
            "peak_ram_mb": round(peak_ram, 1)
        }
        ablation_results.append(res_dict)

        del X, y, X_train, y_train, X_val, y_val, matcher
        force_garbage_collection()

    # 5. Output CSV and MD
    out_csv = "reports/adaptive_blocking_ablation_15k.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "experiment", "blocking_recall", "avg_candidates", "median_candidates",
            "max_candidates", "total_pairs", "precision", "recall", "macro_f0_5",
            "singleton_accuracy", "runtime_seconds", "peak_ram_mb"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ablation_results:
            writer.writerow(r)

    # Copy to adaptive_evidence_aware_er/reports/
    import shutil
    shutil.copyfile(out_csv, "adaptive_evidence_aware_er/reports/adaptive_blocking_ablation_15k.csv")

    # Write Markdown Table
    out_md = "reports/adaptive_blocking_ablation_15k.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Adaptive Multi-View Blocking Ablation Report (15,000 S1 Entities)\n\n")
        f.write("## 1. Comparative Ablation Matrix\n\n")
        f.write("| Experiment | Blocking Recall | Avg Candidates | Median Candidates | Max Candidates | Total Pairs | Precision | Recall | Macro F0.5 | Singleton Acc | Runtime (s) | Peak RAM (MB) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in ablation_results:
            f.write(
                f"| **{r['experiment']}** | {r['blocking_recall']*100:.2f}% | {r['avg_candidates']:.2f} | "
                f"{r['median_candidates']:.1f} | {r['max_candidates']} | {r['total_pairs']:,} | "
                f"{r['precision']:.4f} | {r['recall']:.4f} | **{r['macro_f0_5']:.4f}** | "
                f"{r['singleton_accuracy']*100:.2f}% | {r['runtime_seconds']:.1f}s | {r['peak_ram_mb']:.1f} MB |\n"
            )

        f.write("\n## 2. Key Findings & Acceptance Analysis\n\n")
        base_f05 = ablation_results[0]["macro_f0_5"]
        best_exp = max(ablation_results, key=lambda x: x["macro_f0_5"])
        f.write(f"- **Baseline Macro F0.5**: {base_f05:.4f}\n")
        f.write(f"- **Best Experiment**: **{best_exp['experiment']}** with Macro F0.5 = **{best_exp['macro_f0_5']:.4f}**\n")
        f.write(f"- **Delta in F0.5**: {best_exp['macro_f0_5'] - base_f05:+.4f}\n")
        f.write(f"- **Delta in Blocking Recall**: {best_exp['blocking_recall'] - ablation_results[0]['blocking_recall']:+.4%}\n")

    shutil.copyfile(out_md, "adaptive_evidence_aware_er/reports/adaptive_blocking_ablation_15k.md")
    logger.info("Successfully generated ablation reports.")

if __name__ == "__main__":
    run_all_ablations()
