# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Macro F0.5: 0.9645](https://img.shields.io/badge/Macro%20F0.5-0.9645-brightgreen.svg)]()
[![Validation: PASS](https://img.shields.io/badge/Validator-PASS-success.svg)]()

Production-grade, CPU-optimized, high-precision Business Entity Resolution system for the Amazon ML Challenge 2026. Evaluated on macro-averaged $F_{0.5}$ across multi-million entity datasets from the United States, India, and France.

---

## 1. System Architecture

The pipeline processes heterogeneous business records across Source 1 (reference source), Source 2, and Source 3 without creating dense Cartesian products, strictly bounding peak memory to $< 4.8$ GB RAM.

```
                      Raw Multi-Source TSVs (US, India, France)
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │     Fast In-Memory Normalization     │
                     │  - ASCII Fast-Path & Unicode NFKD    │
                     │  - Regex Legal Suffix Elimination    │
                     │  - O(1) Address Token Canonicalizer  │
                     └──────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │   Option A Multi-View Inverted Index │
                     │  - Exact Name & 7-Prefix (Top 25)    │
                     │  - Number + Street Bigrams (Top 15)  │
                     │  - IDF Character 3-Grams (Top 10)    │
                     │  - Pure Mathematical Set Union       │
                     └──────────────────┬───────────────────┘
                                         │  (<= 50 candidates / S1)
                                         ▼
                     ┌──────────────────────────────────────┐
                     │      RapidFuzz C++ Feature Kernel    │
                     │  - 9 Name Similarity Metrics         │
                     │  - 6 Address Overlap & Digit Signals │
                     │  - 4 Joint Multi-View Rule Counts    │
                     └──────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │       XGBoost Decision Forest        │
                     │  - Hist-Gradient Boosted Trees       │
                     │  - GroupShuffleSplit Training        │
                     │  - Calibrated Macro F0.5 Router      │
                     └──────────────────┬───────────────────┘
                                         │
                                         ▼
                        Formatted Challenge Submissions
                         ├── matching_results.tsv
                         └── candidate_pairs.tsv
```

---

## 2. Key Performance Benchmarks

### Scale-Up Progression (Intel Core i5-11320H CPU, 16 GB RAM)

| Scale Stage | S1 Sample | Pairs Evaluated | Blocking Recall | Candidate Reduction | Val Precision | Val Recall | Val Macro F0.5 | Peak Process RAM | Total Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Option A (1K Benchmark)** | 1,000 | 11,880 | **86.70%** | 99.9998% | 0.9840 | 0.9050 | **0.9550** | 1,060 MB | 124.5s |
| **Option A (5K Full)** | 5,000 | 59,591 | **85.67%** | 99.9996% | 0.9853 | 0.9084 | **0.9587** | 1,893 MB | 281.4s |
| **Option A (15K Scale-up)** | 15,000 | 179,225 | **85.97%** | 99.9996% | **0.9896** | **0.9146** | **0.9645** | 1,859 MB | 460.4s |

### Baseline Comparison on 15,000 S1 Entities (Validation Split: 44,722 Pairs)

- **Baseline 1: Exact Name Rule:** Macro Precision: 0.6913, Macro Recall: 0.6042, Macro F0.5: **0.6301**
- **Baseline 2: Weighted String Similarity:** Macro Precision: 0.2625, Macro Recall: 0.9995, Macro F0.5: **0.3059**
- **Proposed XGBoost + Calibrated Router (Threshold 0.54):** Macro Precision: **0.9896**, Macro Recall: **0.9146**, Macro F0.5: **0.9645** (Singleton Accuracy: **100.00%**)

---

## 3. Directory Structure

```
├── config.yaml                     # Central system configuration
├── requirements.txt                # Pinned Python package dependencies
├── run_pipeline.py                 # Master pipeline CLI runner
├── README.md                       # Comprehensive project documentation
├── Documentation_template.md       # Detailed technical methodology report
├── output/                         # Submission TSV files
│   ├── matching_results.tsv        # Scored leaderboard submission
│   └── candidate_pairs.tsv         # Blocking candidate pairs
├── models/                         # Serialized model artifacts
│   ├── xgboost_matcher.joblib      # Trained XGBoost model
│   └── model_metadata.json         # Calibrated threshold & training parameters
├── reports/                        # Empirical analysis and scaling reports
│   ├── dataset_profile.md          # Full dataset profiling summary
│   ├── dataset_profile.json        # Machine-readable dataset profile
│   ├── blocking_experiments.csv    # Blocking benchmark matrix
│   ├── blocking_option_a_5k.txt    # Option A comparison report
│   ├── scaling_analysis.md         # 1K -> 5K -> 15K scaling analysis
│   ├── train_5000_summary.md       # 5K training run summary
│   ├── train_15000_summary.md      # 15K training run summary
│   ├── model_comparison.csv        # Baseline vs. XGBoost evaluation metrics
│   └── error_analysis.md           # False positive and false negative diagnosis
├── src/                            # Modular source code
│   ├── normalization.py            # Text & address normalization
│   ├── blocking.py                 # Option A multi-view blocking engine
│   ├── features.py                 # 19-dimensional RapidFuzz feature kernel
│   ├── model.py                    # XGBoost classifier and baseline rule matchers
│   ├── router.py                   # Calibrated macro F0.5 decision router
│   ├── evaluation.py               # Competition Macro F0.5 evaluation metric
│   ├── data_loader.py              # Low-memory streaming TSV readers
│   ├── training_data.py            # Training set and candidate generation
│   ├── train.py                    # Training & hyperparameter optimization pipeline
│   ├── predict.py                  # Country-by-country test inference
│   ├── submission.py               # TSV submission file writer
│   └── memory_utils.py             # Memory profiling & GC utilities
└── utils/
    └── validate_submission.py      # Official competition validator
```

---

## 4. Quick Start & Execution Guide

### 4.1 Environment Setup
```bash
# 1. Clone the repository and enter workspace
cd Amazon

# 2. Install pinned dependencies
pip install -r requirements.txt
```

### 4.2 Run Unit Tests
```bash
python -m pytest tests/
```

### 4.3 Profile Dataset
```bash
python run_pipeline.py --stage profile
```

### 4.4 Train Model (15,000 S1 Entities)
```bash
python run_pipeline.py --stage train --train-sample 15000
```

### 4.5 Generate Predictions on Test Set
```bash
# Full test set inference across India, US, France
python run_pipeline.py --stage predict

# Or quick verification run with sample limit
python run_pipeline.py --stage predict --test-sample-per-country 200 --test-targets-per-country 10000
```

### 4.6 Validate Submission
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 5. Fair Play & Academic Integrity

- **Zero External Lookups:** The solution uses **NO external APIs, geocoders, web scraping, or third-party databases**.
- **Self-Contained ML:** All features, blocking keys, and models are trained strictly on the provided training files.
- **Model Parameters:** XGBoost Hist GBDT model has $< 500,000$ parameters, well within the 8 Billion parameter competition limit.
- **Licensing:** Uses only Apache 2.0 / MIT licensed software (XGBoost, RapidFuzz, Scikit-learn).
