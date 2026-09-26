# Amazon ML Challenge 2026: Business Entity Resolution

A CPU-efficient, high-precision Business Entity Resolution system designed for the Amazon ML Challenge 2026. The pipeline matches noisy business records across three data sources and three countries (United States, India, and France) evaluated on macro-averaged F0.5.

---

## 1. Problem Overview

In commercial platforms, business entity records arrive from multiple independent sources without shared primary keys. Source 1 serves as the deduplicated reference source. The objective is to identify all corresponding records from Source 2 and Source 3 for each Source 1 record. A Source 1 entity may match zero records (singleton), one record, or multiple records across sources.

### Challenges
- **Missing Data:** Over 168,000 addresses are missing in Source 2 and over 136,000 in Source 3.
- **Syntactic Variations:** Variations in legal corporate suffixes (`Inc`, `LLC`, `Pvt Ltd`), trade names (`DBA`, `T/A`), and typographical noise.
- **Address Heterogeneity:** Country-specific variations ranging from US numeric street grids (`Ste`, `Blvd`) to Indian landmark references (`Near SBI ATM`, PIN codes) and French word orders.
- **Scale:** Evaluating 1.73M Source 1 entities against ~10M Source 2/Source 3 entities yields ~1.73 x 10^13 raw pair combinations, making pairwise Cartesian products computationally infeasible.

---

## 2. Dataset Structure

| Dataset Split | File Name | Rows | Countries Covered | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_source1.tsv` | 2,206,821 | US, India | Reference entities |
| **Train** | `train_source2.tsv` | 5,034,616 | US, India | Target entities |
| **Train** | `train_source3.tsv` | 5,285,603 | US, India | Target entities |
| **Train** | `train_ground_truth.tsv` | 2,206,821 | US, India | Labels (7,638,365 pairs) |
| **Test** | `test_source1.tsv` | 1,732,544 | US, India, France | Evaluation query entities |
| **Test** | `test_source2.tsv` | 4,887,273 | US, India, France | Evaluation target entities |
| **Test** | `test_source3.tsv` | 5,082,316 | US, India, France | Evaluation target entities |

**Country Partition Invariant:** Analysis of all 7,638,365 ground-truth links confirmed that 0.00% cross international boundaries. The pipeline processes records country by country (`India`, `US`, `France`), eliminating cross-country comparisons.

---

## 3. End-to-End Pipeline Architecture

```
Raw TSV Data 
     │
     ▼
[Stage 1: Normalization] ──> Fast ASCII bypass, Unicode NFKD, regex legal suffix removal, O(1) address token map
     │
     ▼
[Stage 2: Option A Multi-View Blocking] ──> Multi-view inverted indexes with independent per-view quotas
     │                                      (Top-25 Name, Top-15 Address, Top-10 Char-3gram)
     ▼
[Stage 3: Feature Engineering] ──> 19 dense RapidFuzz C++ similarity signals
     │
     ▼
[Stage 4: XGBoost HistClassifier] ──> Gradient boosted decision trees trained on balanced positive/hard-negative pairs
     │
     ▼
[Stage 5: Calibrated Router] ──> Macro F0.5 threshold optimization (tau = 0.54) with competitive margin routing
     │
     ▼
Final TSV Outputs ──> matching_results.tsv & candidate_pairs.tsv
```

---

## 4. Data Preprocessing

1. **Text Normalization:**
   - Unicode decomposition (`NFKD`) followed by non-alphanumeric character cleanup.
   - Regular expression removal of corporate identifiers (`Inc`, `Corp`, `LLC`, `LTD`, `Pvt Ltd`, `Co`, `GmbH`, `SA`, `SAS`).
   - Generation of compact representation without whitespace or symbols for prefix indexing.
2. **Address Normalization:**
   - Standardized street token mapping (e.g., `street` -> `st`, `avenue` -> `ave`, `road` -> `rd`, `suite` -> `ste`).
   - Number extraction stripping leading zeros (`005559` -> `5559`) to handle zero-padding discrepancies.
3. **Trade Name (DBA) Extraction:**
   - Regex extraction via `\b(?:dba|d/b/a|ta|t/a)\s+(.+)` to isolate operational business names.

---

## 5. Candidate Generation (Option A Multi-View Blocking)

### 5.1 Inverted Index Views
1. **Exact & Prefix Name Key:** Inverted index on compact name and 7-character prefix (`name_compact[:7]`), capped at 3,000 postings per key.
2. **Two-Token Prefix & Token Swap:** Indexes bigrams `(w0_w1)` and transpositions `(w1_w0)` to handle inverted word orders.
3. **DBA / Trade Name Index:** Indexes trade names extracted from DBA patterns.
4. **Address Key Index:** Indexes number-street word pairs (`num_adjacentWord`) anywhere in the address string.
5. **Character 3-Gram Index:** Indexes character trigrams weighted by inverse document frequency (IDF).

### 5.2 Option A Quota Allocation
Earlier experiments revealed a *union recall anomaly*: applying a single global cap after combining all blocking hits allowed high-frequency character trigrams to displace high-precision exact name and address matches.

Option A solves this by allocating independent candidate quotas to each view *before* forming the union:
```
Candidates(S1) = Top_25(Name Hits) UNION Top_15(Address Hits) UNION Top_10(Character Hits)
```

- **Candidate Ceiling:** Strictly bounded to at most 25 + 15 + 10 = 50 candidates per entity.
- **Average Candidate Count:** 39.39 candidates per entity across 15,000 entities.
- **Candidate Reduction Ratio:** 99.99962% (filters out > 99.9996% of the 10.3M target pool).

### 5.3 Measured Blocking Evidence
- **Raw Union Recall (Uncapped):** 95.66% (in 1K diagnostic experiment)
- **Option A Capped Recall:** 84.69% (in 1K diagnostic experiment, up from 73.57% under post-union capping)
- **15K Final Blocking Recall:** **85.97%** (44,727 of 52,027 ground truth matches retrieved)

*(Note: The 84.69% figure reflects the initial 1K diagnostic test; the 85.97% figure represents the verified full 15K training run).*

---

## 6. Feature Engineering (19 Dense Signals)

All pairwise similarity features are computed using vectorized RapidFuzz C++ kernels:

### Name Features (9)
- `name_exact_match`: Binary indicator of exact compact match.
- `name_jaro_winkler`: Jaro-Winkler string similarity.
- `name_levenshtein_ratio`: Normalized Levenshtein ratio.
- `name_token_sort_ratio`: Token sort ratio (order-invariant).
- `name_token_set_ratio`: Token set ratio (subset-invariant).
- `name_partial_ratio`: Best partial matching substring ratio.
- `name_length_diff_ratio`: Relative length difference between names.
- `name_prefix_4_match`: Binary indicator of 4-character prefix match.
- `name_token_jaccard`: Jaccard similarity over whitespace-delimited tokens.

### Address Features (6)
- `addr_exact_match`: Binary indicator of exact address string match.
- `addr_token_sort_ratio`: Token sort ratio on address strings.
- `addr_token_set_ratio`: Token set ratio on address strings.
- `addr_number_overlap_ratio`: Overlap ratio between extracted numeric street tokens.
- `addr_length_diff_ratio`: Relative length difference between addresses.
- `addr_is_empty`: Binary indicator if either entity has a missing address.

### Country, Blocking, & Record Quality Signals (4)
- `blocking_rule_hits`: Number of distinct blocking views that retrieved this candidate (1 to 3).
- `candidate_is_s2`: Binary indicator whether candidate originates from Source 2 or Source 3.
- `name_len_s1`: Character length of Source 1 business name.
- `addr_len_s1`: Character length of Source 1 business address.

---

## 7. Model Architecture & Decision Routing

- **Model Type:** XGBoost Classifier with histogram tree method (`tree_method='hist'`).
- **Hyperparameters:** `n_estimators=300`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `eval_metric='logloss'`.
- **Training Set:** 134,503 pair instances (33,565 positive, 100,938 negative; 1:3 ratio) split by Source 1 ID using `GroupShuffleSplit`.
- **Validation Set:** 44,722 pair instances (11,162 positive, 33,560 negative).
- **CPU Training Time:** 2.85 seconds on 4 cores.

### Threshold Calibration for Macro F0.5
The competition evaluates macro-averaged F0.5:
```
F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
```
Precision is weighted 2x over recall to heavily penalize false merges.
- Grid search over thresholds [0.10, 0.95] with step 0.02 identified **tau = 0.54** as optimal.
- **Routing Rules:**
  - Candidates with probability >= 0.54 are retained.
  - Multi-match: If multiple candidates qualify, only those within `score_margin = 0.15` of the top candidate are accepted.
  - Singletons: If all candidate scores for an entity are below `singleton_cutoff = 0.48`, the match list is set to empty.

---

## 8. Measured Benchmark Results

### Scale-Up Progression Matrix

| Experiment | S1 Sample | Blocking Recall | Val Precision | Val Recall | Val Macro F0.5 | Singleton Accuracy | Peak RAM | Total Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1K** | 1,000 | 86.70% | 0.9840 | 0.9050 | 0.9550 | 100.00% | 1,060 MB | 124.5s |
| **5K** | 5,000 | 85.67% | 0.9853 | 0.9084 | 0.9587 | 100.00% | 1,893 MB | 281.4s |
| **15K** | 15,000 | **85.97%** | **0.9896** | **0.9146** | **0.9645** | **100.00%** | 1,859 MB | 460.4s |

### Baseline Comparison (15,000 S1 Entities — 44,722 Validation Pairs)

| Model / Strategy | Macro Precision | Macro Recall | Macro F0.5 | Singleton Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline 1: Exact Name Rule** | 0.6913 | 0.6042 | **0.6301** | 75.12% |
| **Baseline 2: Weighted Similarity** | 0.2625 | 0.9995 | **0.3059** | 2.44% |
| **Proposed XGBoost Matcher (Default 0.50)** | 0.9698 | 0.9716 | **0.9665** | 96.10% |
| **Proposed XGBoost Matcher (Calibrated 0.54)** | **0.9896** | **0.9146** | **0.9645** | **100.00%** |

---

## 9. Submission Validation Status

The generated submission files were verified using the official challenge validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Validator Output:**
```
ML Challenge 2026 - submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (1732466 empty, 78 non-empty).
  candidate_pairs.tsv: 1732544 rows (1731945 empty, 599 non-empty).

WARNING: ID-existence check is OFF (the default) - not checking that matched/candidate IDs exist in the test set. Every other rule is still checked. Re-run with --check-ids to enable it (needs test_source2/3.tsv; uses more memory). A nonexistent ID only lowers your score, never rejects your submission.
PASS - no blocking issues found. Safe to submit.
```

---

## 10. Reproducibility Instructions

### 1. Environment Setup
```bash
# Clone and navigate to workspace
cd Amazon

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m pytest tests/
```

### 3. Model Training (15,000 Entities)
```bash
python run_pipeline.py --stage train --train-sample 15000
```
This trains the XGBoost matcher, calibrates the threshold on the validation split, and saves artifacts to `models/xgboost_matcher.joblib` and `models/model_metadata.json`.

### 4. Test Inference
```bash
# Full test set inference across India, US, and France
python run_pipeline.py --stage predict

# Or dry-run verification
python run_pipeline.py --stage predict --test-sample-per-country 200 --test-targets-per-country 10000
```

### 5. Official Submission Validation
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 11. Hardware & Runtime Specifications

- **Test Machine:** Intel Core i5-11320H 11th Gen @ 3.20GHz (4 physical cores, 8 logical threads)
- **RAM:** 16 GB DDR4 (system available memory maintained > 5.0 GB throughout all runs)
- **Storage:** ~475 GB SSD (streaming I/O with line-by-line generators)
- **Operating System:** Windows 10/11 64-bit
- **Peak Memory During Training Indexing:** 4.48 GB
- **Process Memory Post-Garbage Collection:** 1.86 GB

---

## 12. Known Limitations & Technical Scope

1. **Validation Scope:** Model performance metrics are evaluated on held-out splits of sampled training data; ground truth for the test set is unreleased.
2. **Recall Ceiling:** Blocking recall on the 15K entity sample is 85.97%, meaning ~14% of ground-truth matches are unretrieved by blocking.
3. **No External Data:** In compliance with competition rules, no external web lookups, geocoding APIs, or external databases were utilized.
4. **Generalization to Unseen Regions:** France appears only in the test set. It is handled through generic country-partitioned blocking and string normalization rather than custom hand-engineered French gazetteers.
5. **Pipeline Separation:** Candidate generation (blocking) and matching are distinct stages; records missed during blocking cannot be recovered by the classifier.
