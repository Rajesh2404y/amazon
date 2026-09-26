# Training Run Summary (1,000 S1 Entities)

## 1. Dataset & Indexing
- **Source 1 Sample**: 1,000
- **Source 2 Scanned**: 5,034,616
- **Source 3 Scanned**: 5,285,603
- **Relevant Target Records Indexed**: 1,278,776
- **Ground Truth Matches in Sample**: 3,526

## 2. Blocking Performance
- **Total Candidates Generated**: 17,855
- **Average Candidates / S1**: 17.86
- **Median Candidates / S1**: 15.0
- **Maximum Candidates / S1**: 35
- **P90 / P95 / P99 Candidates**: 35.0 / 35.0 / 35.0
- **Retrieved GT Matches**: 2,696
- **Blocking Recall**: 76.46%
- **Candidate Reduction Ratio**: 99.99980%

## 3. Training & Pairs
- **Positive Pairs**: 2,696
- **Hard Negative Pairs**: 7,081
- **Total Training Pairs**: 9,777
- **Number of Features**: 19
- **Model Training Time**: 1.49s

## 4. Model Evaluation (Macro F0.5)
- **Optimal Decision Threshold**: 0.1
- **Macro Precision**: 0.9683
- **Macro Recall**: 0.9185
- **Macro F0.5 Score**: 0.9453
- **Singleton Accuracy**: 84.61%

## 5. System Performance
- **Peak Process RAM**: 1060.9 MB
- **Minimum Available RAM**: 4.78 GB
- **Total Pipeline Runtime**: 60.41s
