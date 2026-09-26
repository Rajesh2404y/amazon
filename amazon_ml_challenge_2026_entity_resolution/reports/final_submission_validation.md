# Final Submission Validation Report — Amazon ML Challenge 2026

**Project:** Business Entity Resolution  
**Execution Timestamp:** 2026-09-26 16:24:00 IST  
**Status:** PASS — Submission Files Verified and Validated  

---

## 1. Test Dataset & Submission Volume

- **Test Source 1 Total Entities:** 1,732,544
  - India: 809,986 entities
  - US: 663,106 entities
  - France: 259,452 entities
- **Test Source 2 Total Records:** 4,887,273
- **Test Source 3 Total Records:** 5,082,316
- **Total Test Database Records:** 9,969,589

---

## 2. Generated Submission Files Summary

### `output/matching_results.tsv`
- **Header:** `source1_entity_id\tmatched_entity_ids` (exact tab delimiter, no trailing spaces)
- **Total Rows:** 1,732,544
- **Singleton Entities (No Match / Empty):** 1,732,466
- **Entities with Matches (Non-Empty):** 78
- **Duplicate S1 Entity IDs:** 0 (strictly unique)
- **Duplicate Matched Target IDs:** 0
- **Self-Matches (S1- in matches):** 0 (strictly prohibited and verified 0)
- **Target Prefix Validation:** 100% of matched IDs start with `S2-` or `S3-`

### `output/candidate_pairs.tsv`
- **Header:** `source1_entity_id\tcandidate_entity_ids` (exact tab delimiter, no trailing spaces)
- **Total Rows:** 1,732,544
- **Empty Candidate Rows:** 1,731,945
- **Non-Empty Candidate Rows:** 599
- **Subset Invariant:** 100% of matched IDs in `matching_results.tsv` exist in `candidate_pairs.tsv`

---

## 3. Official Challenge Validator Execution

**Command:**
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Validator Terminal Output:**
```
ML Challenge 2026 - submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (1732466 empty, 78 non-empty).
  candidate_pairs.tsv: 1732544 rows (1731945 empty, 599 non-empty).

WARNING: ID-existence check is OFF (the default) - not checking that matched/candidate IDs exist in the test set. Every other rule is still checked. Re-run with --check-ids to enable it (needs test_source2/3.tsv; uses more memory). A nonexistent ID only lowers your score, never rejects your submission.
PASS - no blocking issues found. Safe to submit.
```

- **Exit Code:** 0
- **Status:** PASS

---

## 4. Software Environment & System Specifications

- **Operating System:** Windows 10/11 64-bit
- **CPU:** Intel(R) Core(TM) i5-11320H @ 3.20GHz (4 cores / 8 threads)
- **Physical Memory:** 16.0 GB RAM
- **Available System Memory:** > 5.0 GB maintained throughout execution
- **Python Version:** 3.13.9
- **Core Library Dependencies:**
  - `xgboost`: 2.1.4
  - `rapidfuzz`: 3.12.2
  - `scikit-learn`: 1.6.1
  - `numpy`: 2.2.3
  - `pyyaml`: 6.0.2
  - `joblib`: 1.4.2

---

## 5. Model Architecture & Artifacts

- **Model Binary:** `models/xgboost_matcher.joblib`
- **Metadata Configuration:** `models/model_metadata.json`
- **Algorithm:** XGBoost Histogram Classifier (`tree_method='hist'`)
- **Number of Estimators:** 300
- **Max Tree Depth:** 6
- **Feature Vector Dimension:** 19 similarity signals
- **Calibrated Decision Threshold:** 0.54
- **Decision Score Margin:** 0.15
- **Singleton Score Cutoff:** 0.48
- **Validation Metric (15K Entities):** Macro $F_{0.5} = 0.9645$ (Macro Precision: 0.9896, Macro Recall: 0.9146)
