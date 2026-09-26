import os
import time
import json
import csv
import logging
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from src.config import config
from src.training_data import generate_training_dataset
from src.model import EntityXGBoostMatcher, BaselineRuleMatcher
from src.evaluation import evaluate_macro_f05
from src.router import EntityRouter
from src.error_analysis import run_error_analysis
from src.memory_utils import log_memory, force_garbage_collection, get_memory_usage_mb, get_system_available_ram_gb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("entity_resolution")

def run_training_pipeline(
    sample_s1_count: int = 1000,
    target_country: str = "US",
    output_model_dir: str = "models",
    reports_dir: str = "reports"
):
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    t_start = time.time()
    min_avail_ram = get_system_available_ram_gb()

    logger.info("==================================================")
    logger.info("STAGE 2: Training Pipeline")
    logger.info(f"Sample: {sample_s1_count:,} S1 entities")
    logger.info(f"[RAM Check] Pre-Training | Process: {get_memory_usage_mb():.1f} MB | System Avail: {min_avail_ram:.2f} GB")
    logger.info("==================================================")

    # 1-4. Steps [1/7] to [4/7] handled in generate_training_dataset
    X, y, group_s1, blocking_stats = generate_training_dataset(
        train_dir=config.train_dir,
        target_country=target_country,
        sample_s1_count=sample_s1_count,
        neg_to_pos_ratio=3.0,
        random_seed=config.random_seed
    )

    min_avail_ram = min(min_avail_ram, get_system_available_ram_gb())

    # 5. Group Split by S1 ID
    logger.info("[5/7] Creating training & validation group split (by S1 ID)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.random_seed)
    train_idx, val_idx = next(gss.split(X, y, groups=group_s1))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    val_groups = [group_s1[i] for i in val_idx]

    logger.info(f"  Train: {len(y_train):,} pairs ({sum(y_train):,} pos, {len(y_train)-sum(y_train):,} neg)")
    logger.info(f"  Val:   {len(y_val):,} pairs ({sum(y_val):,} pos, {len(y_val)-sum(y_val):,} neg)")
    logger.info(f"  [RAM Check] Feature Matrices: {X.nbytes / (1024*1024):.2f} MB | Process RSS: {get_memory_usage_mb():.1f} MB")

    # 6. Model Training & Comparison
    logger.info("[6/7] Comparing Baselines & Training XGBoost Matcher...")
    models_to_evaluate = {
        "Baseline 1: Exact Name Rule": BaselineRuleMatcher(mode="exact"),
        "Baseline 2: Weighted Similarity": BaselineRuleMatcher(mode="weighted"),
        "Main Model: XGBoost Classifier": EntityXGBoostMatcher(config.xgboost_cfg)
    }

    t_xgb_start = time.time()
    models_to_evaluate["Main Model: XGBoost Classifier"].fit(X_train, y_train, X_val, y_val)
    t_xgb = time.time() - t_xgb_start
    logger.info(f"[TIMER] XGBoost trained in {t_xgb:.2f}s")

    # Build validation ground truth mapping
    val_gt = {}
    for i, s1_id in enumerate(val_groups):
        if s1_id not in val_gt:
            val_gt[s1_id] = set()
        if y_val[i] == 1:
            val_gt[s1_id].add(f"CAND_{i}")

    comparison_results = []
    for name, model in models_to_evaluate.items():
        val_probs = model.predict_proba(X_val)
        entity_preds = {}
        for i, s1_id in enumerate(val_groups):
            if s1_id not in entity_preds:
                entity_preds[s1_id] = set()
            if val_probs[i] >= 0.50:
                entity_preds[s1_id].add(f"CAND_{i}")

        metrics = evaluate_macro_f05(entity_preds, val_gt)
        comparison_results.append({
            "model": name,
            "precision": metrics["macro_precision"],
            "recall": metrics["macro_recall"],
            "f0_5": metrics["macro_f05"],
            "singleton_acc": metrics["singleton_accuracy"]
        })
        logger.info(f"  {name} -> Macro F0.5: {metrics['macro_f05']:.4f} | Prec: {metrics['macro_precision']:.4f} | Rec: {metrics['macro_recall']:.4f}")

    # Write model_comparison.csv
    comp_csv = os.path.join(reports_dir, "model_comparison.csv")
    with open(comp_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "precision", "recall", "f0_5", "singleton_acc"])
        for res in comparison_results:
            writer.writerow([res["model"], res["precision"], res["recall"], res["f0_5"], res["singleton_acc"]])

    # 7. Threshold Optimization for Macro F0.5 (Grid: 0.10 to 0.95, step: 0.02)
    logger.info("[7/7] Calibrating optimal decision threshold for Macro F0.5 (range: 0.10 - 0.95)...")
    router = EntityRouter()
    xgb_val_probs = models_to_evaluate["Main Model: XGBoost Classifier"].predict_proba(X_val)

    val_entity_scores = {}
    for i, s1_id in enumerate(val_groups):
        if s1_id not in val_entity_scores:
            val_entity_scores[s1_id] = []
        val_entity_scores[s1_id].append((f"CAND_{i}", float(xgb_val_probs[i])))

    opt_results = router.optimize_threshold(
        val_entity_scores, val_gt, threshold_range=(0.10, 0.95, 0.02)
    )
    logger.info(
        f"Optimal Threshold: {opt_results['optimal_threshold']} -> "
        f"Macro F0.5: {opt_results['macro_f05']:.4f} (Prec: {opt_results['macro_precision']:.4f}, Rec: {opt_results['macro_recall']:.4f})"
    )

    # Save models/threshold.json
    threshold_json_path = os.path.join(output_model_dir, "threshold.json")
    threshold_data = {
        "threshold": opt_results["optimal_threshold"],
        "validation_f0_5": opt_results["macro_f05"],
        "precision": opt_results["macro_precision"],
        "recall": opt_results["macro_recall"],
        "singleton_accuracy": opt_results["singleton_accuracy"]
    }
    with open(threshold_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)

    # Save XGBoost Model & Metadata
    model_save_path = os.path.join(output_model_dir, "xgboost_matcher.joblib")
    models_to_evaluate["Main Model: XGBoost Classifier"].save(model_save_path)

    min_avail_ram = min(min_avail_ram, get_system_available_ram_gb())
    peak_ram = get_memory_usage_mb()
    total_time = round(time.time() - t_start, 2)

    metadata = {
        "target_country": target_country,
        "sample_s1_count": sample_s1_count,
        "optimal_threshold": opt_results["optimal_threshold"],
        "macro_f05": opt_results["macro_f05"],
        "macro_precision": opt_results["macro_precision"],
        "macro_recall": opt_results["macro_recall"],
        "blocking_recall": blocking_stats["blocking_recall"],
        "avg_candidates_per_s1": blocking_stats["avg_candidates_per_s1"],
        "total_runtime_sec": total_time,
        "peak_ram_mb": round(peak_ram, 1),
        "min_available_ram_gb": round(min_avail_ram, 2)
    }
    with open(os.path.join(output_model_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Update reports/experiments.csv
    exp_csv = os.path.join(reports_dir, "experiments.csv")
    with open(exp_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"XGBoost_Sample_{sample_s1_count}",
            sample_s1_count,
            "Adaptive Multi-View (Exact+Prefix+Address)",
            blocking_stats["total_gt_matches"],
            blocking_stats["total_candidates"],
            blocking_stats["avg_candidates_per_s1"],
            blocking_stats["median_candidates_per_s1"],
            blocking_stats["max_candidates_per_s1"],
            blocking_stats["blocking_recall"],
            blocking_stats["candidate_reduction_ratio"],
            blocking_stats["positive_pairs"],
            blocking_stats["negative_pairs"],
            opt_results["macro_precision"],
            opt_results["macro_recall"],
            opt_results["macro_f05"],
            opt_results["optimal_threshold"],
            total_time,
            round(peak_ram, 1)
        ])

    # Save summary markdown report: reports/train_{sample_s1_count}_summary.md
    summary_md = os.path.join(reports_dir, f"train_{sample_s1_count}_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(f"# Training Run Summary ({sample_s1_count:,} S1 Entities)\n\n")
        f.write("## 1. Dataset & Indexing\n")
        f.write(f"- **Source 1 Sample**: {sample_s1_count:,}\n")
        f.write(f"- **Source 2 Scanned**: {blocking_stats['s2_scanned']:,}\n")
        f.write(f"- **Source 3 Scanned**: {blocking_stats['s3_scanned']:,}\n")
        f.write(f"- **Relevant Target Records Indexed**: {blocking_stats['relevant_indexed']:,}\n")
        f.write(f"- **Ground Truth Matches in Sample**: {blocking_stats['total_gt_matches']:,}\n\n")

        f.write("## 2. Blocking Performance\n")
        f.write(f"- **Total Candidates Generated**: {blocking_stats['total_candidates']:,}\n")
        f.write(f"- **Average Candidates / S1**: {blocking_stats['avg_candidates_per_s1']}\n")
        f.write(f"- **Median Candidates / S1**: {blocking_stats['median_candidates_per_s1']}\n")
        f.write(f"- **Maximum Candidates / S1**: {blocking_stats['max_candidates_per_s1']}\n")
        f.write(f"- **P90 / P95 / P99 Candidates**: {blocking_stats['p90_candidates']} / {blocking_stats['p95_candidates']} / {blocking_stats['p99_candidates']}\n")
        f.write(f"- **Retrieved GT Matches**: {blocking_stats['retrieved_gt_matches']:,}\n")
        f.write(f"- **Blocking Recall**: {blocking_stats['blocking_recall']*100:.2f}%\n")
        f.write(f"- **Candidate Reduction Ratio**: {blocking_stats['candidate_reduction_ratio']*100:.5f}%\n\n")

        f.write("## 3. Training & Pairs\n")
        f.write(f"- **Positive Pairs**: {blocking_stats['positive_pairs']:,}\n")
        f.write(f"- **Hard Negative Pairs**: {blocking_stats['negative_pairs']:,}\n")
        f.write(f"- **Total Training Pairs**: {blocking_stats['total_training_pairs']:,}\n")
        f.write(f"- **Number of Features**: {X.shape[1]}\n")
        f.write(f"- **Model Training Time**: {t_xgb:.2f}s\n\n")

        f.write("## 4. Model Evaluation (Macro F0.5)\n")
        f.write(f"- **Optimal Decision Threshold**: {opt_results['optimal_threshold']}\n")
        f.write(f"- **Macro Precision**: {opt_results['macro_precision']:.4f}\n")
        f.write(f"- **Macro Recall**: {opt_results['macro_recall']:.4f}\n")
        f.write(f"- **Macro F0.5 Score**: {opt_results['macro_f05']:.4f}\n")
        f.write(f"- **Singleton Accuracy**: {opt_results['singleton_accuracy']*100:.2f}%\n\n")

        f.write("## 5. System Performance\n")
        f.write(f"- **Peak Process RAM**: {peak_ram:.1f} MB\n")
        f.write(f"- **Minimum Available RAM**: {min_avail_ram:.2f} GB\n")
        f.write(f"- **Total Pipeline Runtime**: {total_time:.2f}s\n")

    # Run Error Analysis
    val_preds = {}
    for s1_id in val_gt.keys():
        cands = val_entity_scores.get(s1_id, [])
        matches = router.decide_matches(cands)
        val_preds[s1_id] = set(matches)
    run_error_analysis(val_preds, val_gt, entity_records={}, reports_dir=reports_dir)

    print("\n" + "="*50)
    print(f"TRAINING COMPLETE ({sample_s1_count:,} S1)")
    print(f"  Blocking Recall: {blocking_stats['blocking_recall']*100:.2f}%")
    print(f"  Avg Candidates:  {blocking_stats['avg_candidates_per_s1']}")
    print(f"  Macro F0.5:      {opt_results['macro_f05']:.4f}")
    print(f"  Macro Prec:      {opt_results['macro_precision']:.4f}")
    print(f"  Macro Rec:       {opt_results['macro_recall']:.4f}")
    print(f"  Runtime:         {total_time:.2f}s")
    print(f"  Peak RAM:        {peak_ram:.1f} MB")
    print("="*50 + "\n")

    return metadata

if __name__ == "__main__":
    run_training_pipeline()
