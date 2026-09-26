# Training Run Summary (15,000 S1 Entities — Option A Multi-View Blocking)

## 1. Dataset & Indexing
- **Source 1 Sample**: 15,000 entities
- **Source 2 Scanned**: 5,034,616 records
- **Source 3 Scanned**: 5,285,603 records
- **Relevant Target Records Indexed**: 3,069,730
- **Ground Truth Matches in Sample**: 52,027 (unique target IDs)
- **Target Index Build Time**: 202.07s

## 2. Blocking Performance (Option A Per-View Quotas)
- **Total Candidates Generated**: 590,790
- **Average Candidates / S1**: 39.39
- **Median Candidates / S1**: 42.0
- **Maximum Candidates / S1**: 50
- **Retrieved GT Matches**: 44,727 (out of 52,027)
- **Blocking Recall**: **85.97%**
- **Candidate Reduction Ratio**: **99.99962%**
- **Candidate & Feature Generation Time**: 247.87s

## 3. Training & Pairs
- **Positive Pairs**: 44,727
- **Hard Negative Pairs**: 134,498
- **Total Training Pairs**: 179,225
- **Train Set (75%)**: 134,503 pairs (33,565 pos, 100,938 neg)
- **Validation Set (25%)**: 44,722 pairs (11,162 pos, 33,560 neg)
- **Feature Matrix RAM**: 12.99 MB
- **Model Training Time**: 2.85s

## 4. Model Evaluation (Macro F0.5)
- **Baseline 1 (Exact Name Rule)**: Precision: 0.6913, Recall: 0.6042, Macro F0.5: **0.6301**
- **Baseline 2 (Weighted Similarity)**: Precision: 0.2625, Recall: 0.9995, Macro F0.5: **0.3059**
- **Main Model (XGBoost Default 0.50)**: Precision: 0.9698, Recall: 0.9716, Macro F0.5: **0.9665**
- **Calibrated Optimal Threshold**: **0.54**
- **Final Calibrated Macro F0.5**: **0.9645**
  - **Macro Precision**: **0.9896**
  - **Macro Recall**: **0.9146**
  - **Singleton Accuracy**: 100.00%

## 5. System Resource Utilization
- **Indexing RAM Peak**: ~4.48 GB
- **Candidate Generation RAM**: ~4.73 GB
- **Process RSS Post-Collection**: 1,859.0 MB (~1.86 GB)
- **Available System RAM**: > 5.0 GB maintained throughout
- **Total Pipeline Runtime**: 460.41s (~7.67 minutes)
