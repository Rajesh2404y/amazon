#!/usr/bin/env python3
"""
Amazon ML Challenge 2026: Business Entity Resolution
Master Pipeline CLI Runner.
"""

import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("entity_resolution")

def stage_profile():
    logger.info("==================================================")
    logger.info("STAGE 1: Dataset Profiling")
    logger.info("==================================================")
    from src.dataset_profile import run_profiling
    run_profiling()

def stage_train(sample_size: int = 50000):
    logger.info("==================================================")
    logger.info(f"STAGE 2: Training Pipeline (Sample: {sample_size:,} S1 entities)")
    logger.info("==================================================")
    from src.train import run_training_pipeline
    run_training_pipeline(sample_s1_count=sample_size)

def stage_blocking(sample_size: int = 10000):
    logger.info("==================================================")
    logger.info(f"STAGE: Multi-View Blocking Evaluation (Sample: {sample_size:,})")
    logger.info("==================================================")
    from src.training_data import generate_training_dataset
    from src.config import config
    _, _, _, stats = generate_training_dataset(
        train_dir=config.train_dir,
        target_country="US",
        sample_s1_count=sample_size,
        neg_to_pos_ratio=1.0
    )
    logger.info(f"Blocking Evaluation Results:\n"
                f"  Recall: {stats['blocking_recall']*100:.2f}%\n"
                f"  Average Candidates per S1: {stats['avg_candidates_per_s1']}")

def stage_predict(max_s1_per_country: int = None, max_targets_per_country: int = None):
    logger.info("==================================================")
    logger.info("STAGE 3: Test Inference Pipeline")
    logger.info("==================================================")
    from src.predict import run_test_inference
    run_test_inference(max_s1_per_country=max_s1_per_country, max_targets_per_country=max_targets_per_country)

def stage_validate():
    logger.info("==================================================")
    logger.info("STAGE 4: Validating Submission")
    logger.info("==================================================")
    cmd = [
        sys.executable,
        os.path.join("utils", "validate_submission.py"),
        "--matching", os.path.join("output", "matching_results.tsv"),
        "--candidate", os.path.join("output", "candidate_pairs.tsv"),
        "--test-dir", os.path.join("dataset", "test")
    ]
    logger.info(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error(f"Validation FAILED with exit code {res.returncode}")
        sys.exit(res.returncode)
    else:
        logger.info("Validation PASSED successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026: Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["profile", "blocking", "train", "evaluate", "predict", "validate", "all"],
        default="all",
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument(
        "--train-sample",
        type=int,
        default=50000,
        help="Number of Source 1 entities to sample for training (default: 50,000)"
    )
    parser.add_argument(
        "--test-sample-per-country",
        type=int,
        default=None,
        help="Optional test sample limit per country for quick dry runs"
    )
    parser.add_argument(
        "--test-targets-per-country",
        type=int,
        default=None,
        help="Optional test target limit per country for quick dry runs"
    )
    args = parser.parse_args()

    if args.stage == "profile":
        stage_profile()
    elif args.stage == "blocking":
        stage_blocking()
    elif args.stage == "train":
        stage_train(sample_size=args.train_sample)
    elif args.stage == "evaluate":
        logger.info("Evaluating model artifacts from reports/model_comparison.csv...")
        import pandas as pd
        if os.path.exists("reports/model_comparison.csv"):
            df = pd.read_csv("reports/model_comparison.csv")
            print(df.to_string(index=False))
        else:
            logger.warning("reports/model_comparison.csv not found. Run --stage train first.")
    elif args.stage == "predict":
        stage_predict(
            max_s1_per_country=args.test_sample_per_country,
            max_targets_per_country=args.test_targets_per_country
        )
    elif args.stage == "validate":
        stage_validate()
    elif args.stage == "all":
        stage_profile()
        stage_train(sample_size=args.train_sample)
        stage_predict(
            max_s1_per_country=args.test_sample_per_country,
            max_targets_per_country=args.test_targets_per_country
        )
        stage_validate()

if __name__ == "__main__":
    main()
