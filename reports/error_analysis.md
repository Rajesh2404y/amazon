# Error Analysis & Model Diagnostics

## 1. Singleton Breakdown
- **True Singletons**: 205
- **Correctly Predicted Singletons**: 199 (97.07%)
- **False Merges on Singletons**: 6 (2.93%)

## 2. False Positive Patterns
Common causes of False Positives observed:
1. Shared commercial buildings/addresses with distinct business names.
2. Chain branches or parent companies with identical brand names across different store locations.
3. Generic business prefixes (e.g. 'Shree', 'Star', 'Global') exceeding similarity thresholds.

### Sample False Positives:
- **S1**: `S1-784078088` (N/A, N/A)
  - Predicted: `['CAND_2245']`
  - Ground Truth: `[]`

- **S1**: `S1-556845820` (N/A, N/A)
  - Predicted: `['CAND_2643']`
  - Ground Truth: `['CAND_2633', 'CAND_2634', 'CAND_2635']`

- **S1**: `S1-998949892` (N/A, N/A)
  - Predicted: `['CAND_2810', 'CAND_2796']`
  - Ground Truth: `['CAND_2795', 'CAND_2792', 'CAND_2793', 'CAND_2794', 'CAND_2791']`

- **S1**: `S1-568551826` (N/A, N/A)
  - Predicted: `['CAND_2813']`
  - Ground Truth: `['CAND_2811', 'CAND_2812']`

- **S1**: `S1-548281154` (N/A, N/A)
  - Predicted: `['CAND_3675']`
  - Ground Truth: `['CAND_3665', 'CAND_3667', 'CAND_3668', 'CAND_3666']`

## 3. False Negative Patterns
Common causes of False Negatives observed:
1. Severe transliteration where Latin script is phonetically translated to Indic scripts without shared English tokens.
2. Missing address in Source 2/3 combined with heavy name typo or DBA trade names.
3. Extreme token shuffling in multi-line municipal addresses.

### Sample False Negatives:
- **S1**: `S1-504790211` (N/A, N/A)
  - Missed IDs: `['CAND_57']`
  - Predicted IDs: `['CAND_60', 'CAND_58', 'CAND_59', 'CAND_56']`

- **S1**: `S1-743171772` (N/A, N/A)
  - Missed IDs: `['CAND_139']`
  - Predicted IDs: `['CAND_138', 'CAND_136', 'CAND_137']`

- **S1**: `S1-424529930` (N/A, N/A)
  - Missed IDs: `['CAND_178']`
  - Predicted IDs: `['CAND_176', 'CAND_177']`

- **S1**: `S1-846529766` (N/A, N/A)
  - Missed IDs: `['CAND_190']`
  - Predicted IDs: `['CAND_188', 'CAND_189']`

- **S1**: `S1-800915873` (N/A, N/A)
  - Missed IDs: `['CAND_217']`
  - Predicted IDs: `['CAND_219', 'CAND_218', 'CAND_220', 'CAND_216']`

