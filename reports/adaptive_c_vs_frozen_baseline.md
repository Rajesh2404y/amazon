# Strict Apples-to-Apples Comparison: Frozen Baseline vs. Adaptive-C

## 1. Experimental Methodology & Invariants

Both pipelines were evaluated on the identical 15,000 Source 1 validation sample using:
- Identical random seed (`42`)
- Identical 75%/25% `GroupShuffleSplit` partitioned by Source 1 ID
- Identical 1:3 positive-to-hard-negative sampling policy
- Identical 19-dimensional RapidFuzz feature extraction kernel
- Identical XGBoost HistClassifier configuration
- Identical threshold calibration search range `(0.10, 0.95, 0.02)` maximizing Macro F0.5
- Identical competition Macro F0.5 evaluation function

## 2. Quantitative Comparison Table

| Metric | Frozen Baseline | Adaptive-C | Delta |
| :--- | :--- | :--- | :--- |
| **Blocking Recall** | 85.92% (44,702 / 52,027) | 91.87% (47,796 / 52,027) | **+5.95%** (+3,094 matches) |
| **Average Candidates / S1** | 39.45 | 42.73 | +3.28 |
| **Median Candidates / S1** | 42.0 | 45.0 | +3.0 |
| **Maximum Candidates / S1** | 50 | 60 | +10 |
| **Total Training Pairs** | 179,136 | 191,273 | +12,137 |
| **Validation Precision** | 0.9888 | 0.9888 | -0.0000 |
| **Validation Recall** | 0.9132 | 0.9239 | +0.0107 |
| **Validation Macro F0.5** | **0.9630** | **0.9671** | **+0.0041** |
| **Optimal Decision Threshold** | 0.10 | 0.10 | +0.00 |
| **Singleton Accuracy** | 95.17% | 96.67% | +1.50% |
| **Pipeline Runtime** | 224.0s | 247.8s | +23.8s |
| **Peak Process RAM** | 6597.6 MB | 6807.2 MB | +209.6 MB |

## 3. Formal Acceptance Determination

### Decision: **ACCEPT**
Adaptive-C achieves Macro F0.5 = **0.9671**, exceeding the frozen baseline (0.9630) by **+0.0041** while lifting blocking recall from 85.92% to 91.87% (+5.95%) with high precision (0.9888) and controlled candidate density (Avg: 42.73).
