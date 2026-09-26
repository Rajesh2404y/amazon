# Adaptive Multi-View Blocking Ablation Report (15,000 S1 Entities)

## 1. Comparative Ablation Matrix

| Experiment | Blocking Recall | Avg Candidates | Median Candidates | Max Candidates | Total Pairs | Precision | Recall | Macro F0.5 | Singleton Acc | Runtime (s) | Peak RAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 85.81% | 39.45 | 42.0 | 50 | 178,901 | 0.9854 | 0.9173 | **0.9623** | 95.12% | 296.1s | 5295.0 MB |
| **Adaptive-A** | 85.98% | 40.41 | 44.0 | 50 | 179,274 | 0.9888 | 0.9201 | **0.9658** | 96.45% | 267.6s | 5360.4 MB |
| **Adaptive-B** | 91.32% | 35.83 | 38.0 | 50 | 189,258 | 0.9882 | 0.9158 | **0.9638** | 94.53% | 285.6s | 5372.5 MB |
| **Adaptive-C** | 91.90% | 42.74 | 45.0 | 60 | 191,342 | 0.9889 | 0.9259 | **0.9680** | 94.61% | 289.8s | 5256.7 MB |
| **Adaptive-D** | 92.53% | 52.98 | 56.0 | 75 | 193,370 | 0.9884 | 0.9232 | **0.9666** | 97.09% | 1727.0s | 5262.2 MB |

## 2. Key Findings & Acceptance Analysis

- **Baseline Macro F0.5**: 0.9623
- **Best Experiment**: **Adaptive-C** with Macro F0.5 = **0.9680**
- **Delta in F0.5**: +0.0057
- **Delta in Blocking Recall**: +6.0900%
