# Dataset Profile Report: Amazon ML Challenge 2026

## 1. Overview
The Amazon ML Challenge 2026 dataset focuses on Business Entity Resolution across three data sources:
- **Source 1 (Reference)**: Clean, deduplicated entity records.
- **Source 2**: Noisy entity records with abbreviations, typographical errors, and partial addresses.
- **Source 3**: Additional noisy entity records with transliterations, legal suffix variations, and web domains.

---

## 2. Dataset Dimensions & Completeness

| File | Total Rows | Unique IDs | Duplicate IDs | Missing Names | Missing Addresses | Missing Country |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 2,206,821 | 2,206,821 | 0 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `train_source2.tsv` | 5,034,616 | 5,034,616 | 0 | 0 (0.00%) | 168,967 (3.35%) | 0 (0.00%) |
| `train_source3.tsv` | 5,285,603 | 5,285,603 | 0 | 0 (0.00%) | 175,916 (3.33%) | 0 (0.00%) |
| `test_source1.tsv` | 1,732,544 | 1,732,544 | 0 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `test_source2.tsv` | 4,887,273 | 4,887,273 | 0 | 0 (0.00%) | 129,408 (2.65%) | 0 (0.00%) |
| `test_source3.tsv` | 5,082,316 | 5,082,316 | 0 | 0 (0.00%) | 136,098 (2.68%) | 0 (0.00%) |

**Total training records**: 12,527,040  
**Total test records**: 11,702,133  

---

## 3. Country Distribution

| Country | `train_source1` | `train_source2` | `train_source3` | `test_source1` | `test_source2` | `test_source3` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **United States (US)** | 1,323,633 (60.0%) | 3,016,817 (59.9%) | 3,170,056 (60.0%) | 663,106 (38.3%) | 1,871,330 (38.3%) | 1,945,701 (38.3%) |
| **India** | 883,188 (40.0%) | 2,017,799 (40.1%) | 2,115,547 (40.0%) | 809,986 (46.7%) | 2,312,565 (47.3%) | 2,405,000 (47.3%) |
| **France** | *Unseen in train* | *Unseen in train* | *Unseen in train* | 259,452 (15.0%) | 703,378 (14.4%) | 731,615 (14.4%) |

### Critical Finding: Cross-Country Matching
Across all **7,638,365** matches in `train_ground_truth.tsv`:
- **Same country matches**: 7,638,365 (100.00%)
- **Cross country matches**: 0 (0.00%)
- **Implication**: Strict country partitioning during blocking and inference is mathematically verified to incur 0 recall loss while drastically reducing search complexity and false positives.

---

## 4. Ground Truth Matching Statistics

- **Total S1 Entities**: 2,206,821
- **Singletons (0 matches)**: 123,247 (**5.58%**)
- **Single Match (1 match)**: 119,157 (**5.40%**)
- **Multiple Matches (>1 matches)**: 1,964,417 (**89.02%**)
- **Total Matched Targets**: 7,638,365
  - Source 2 Targets: 3,693,619 (48.36%)
  - Source 3 Targets: 3,944,746 (51.64%)
- **Average Matches per S1**: 3.461
- **Median Matches per S1**: 3.0
- **Maximum Matches for Single S1**: 11

---

## 5. String Length Characteristics

| Metric | Name Min | Name Max | Name Mean | Name Median | Address Min | Address Max | Address Mean | Address Median |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1` | 3 | 87 | 24.03 | 24.0 | 13 | 256 | 52.05 | 41.0 |
| `train_source2` | 2 | 103 | 25.10 | 25.0 | 0 | 230 | 46.21 | 37.0 |
| `train_source3` | 2 | 96 | 25.21 | 25.0 | 0 | 234 | 46.73 | 42.0 |
| `test_source1` | 3 | 85 | 23.84 | 24.0 | 14 | 238 | 57.13 | 50.0 |
| `test_source2` | 2 | 90 | 25.72 | 25.0 | 0 | 234 | 50.37 | 42.0 |
| `test_source3` | 2 | 99 | 25.66 | 25.0 | 0 | 232 | 48.75 | 43.0 |

---

## 6. Discovered Noise & Variation Archetypes

Inspection of true matched pairs revealed 5 primary noise classes:
1. **Transliteration & Non-Latin Scripts**: Entities registered in regional scripts (Tamil, Devanagari Hindi, Kannada) matching Latin English counterparts (e.g. `Raj Investments LLP` vs `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`).
2. **Missing Address Field**: Approximately 3.3% of S2 and S3 records have completely empty addresses, requiring robust name-only similarity features.
3. **Legal Suffix Evolution**: Additions, deletions, and transformations of legal forms (e.g., `Inc`, `LLC`, `Private Limited` <-> `Pvt Ltd`, `LLP`, or complete omission).
4. **URL & Domain Representation**: Some source records represent companies by their domain name (e.g. `maurewilliamscolombier.com`).
5. **Address Shuffling & Typos**: Street numbers transposed to the end of strings, missing postal codes, or localized abbreviations (`St` vs `Saint`, `Avenue` vs `Ave`, state codes `NY` vs `New York`).
