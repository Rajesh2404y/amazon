# Training Run Summary (5,000 S1 Entities)

## 1. Dataset & Indexing
- **Source 1 Sample**: 5,000
- **Source 2 Scanned**: 5,034,616
- **Source 3 Scanned**: 5,285,603
- **Relevant Target Records Indexed**: 3,437,131
- **Ground Truth Matches in Sample**: 17,318

## 2. Blocking Performance
- **Total Candidates Generated**: 200,495
- **Average Candidates / S1**: 40.1
- **Median Candidates / S1**: 42.0
- **Maximum Candidates / S1**: 50
- **P90 / P95 / P99 Candidates**: 49.0 / 49.0 / 50.0
- **Retrieved GT Matches**: 14,836
- **Blocking Recall**: 85.67%
- **Candidate Reduction Ratio**: 99.99960%

## 3. Training & Pairs
- **Positive Pairs**: 14,836
- **Hard Negative Pairs**: 44,755
- **Total Training Pairs**: 59,591
- **Number of Features**: 19
- **Model Training Time**: 1.20s

## 4. Model Evaluation (Macro F0.5)
- **Optimal Decision Threshold**: 0.56
- **Macro Precision**: 0.9853
- **Macro Recall**: 0.9084
- **Macro F0.5 Score**: 0.9587
- **Singleton Accuracy**: 95.65%

## 5. System Performance
- **Peak Process RAM**: 1893.3 MB
- **Minimum Available RAM**: 5.21 GB
- **Total Pipeline Runtime**: 281.44s
