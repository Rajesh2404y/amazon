# Blocking Miss Diagnostics & Hard Case Analysis

## 1. Executive Summary
Prior to the multi-view blocking enhancements, the baseline system achieved **76.28% blocking recall** on the 5,000 Source 1 sample, missing **4,108 ground truth matches**. 

We performed an in-depth diagnostic audit on the missed pairs. The analysis revealed that false dismissals were not random, but concentrated in 5 identifiable structural archetypes. By designing specialized blocking views for each archetype, we engineered targeted improvements to recover the missing links while maintaining high candidate reduction.

---

## 2. Categorization of Missed Pairs & Engineering Solutions

### Category 1: Address Leading Zeros & Number Shuffling
- **Diagnostic Observation**: Addresses with municipal unit numbers, leading zeroes (e.g. `005559` or `0337`), or addresses where city/state appears before the street number (e.g. `arizona phoenix 1712 montebello avenue`).
- **Real Missed Example**:
  - **S1**: `B+ Retail Inc` | Address: `1712 Montebello Avenue, Phoenix, AZ`
  - **Target (S3-70942743)**: `b services` | Address: `arizona phoenix 1712 montebello avenue`
- **Failure Mode**: The baseline index only inspected `words[0]` for numbers. When state/city preceded the street number, the building number was skipped.
- **Solution View**: Scan for *any* numeric token in the address string, strip leading zeroes (`w.lstrip('0')`), and pair it with both adjacent tokens: `f"{num}_{word}"`.

### Category 2: DBA & Trade Name Differences
- **Diagnostic Observation**: The Source 1 record uses the corporate trade name, while Source 2/3 lists the parent entity followed by `DBA <tradename>` (or vice-versa).
- **Real Missed Example**:
  - **S1**: `Christ Chapel` | Address: `2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD`
  - **Target (S3-183080822)**: `ectosyn dba christ chapel` | Address: `2100 cameron drive unit apartment g dundalk md`
- **Failure Mode**: The prefix index compared the start of the string (`ectosyn...` vs `christ...`), failing to match the DBA suffix.
- **Solution View**: Precompiled regex `\b(?:dba|d/b/a|ta|t/a)\s+(.+)` extracts the DBA name and indexes it in both exact and prefix views.

### Category 3: Typographical OCR Errors in Names
- **Diagnostic Observation**: Single-character substitutions and OCR anomalies (e.g., numeral `1` substituting lowercase `l`, or phonetic letter swaps).
- **Real Missed Example**:
  - **S1**: `Crystal Lending PC` | Address: `11643 Prosperity Road, South Jordan, UT`
  - **Target (S2-482219248)**: `crysta1 lending pc` (`1` instead of `l`)
- **Failure Mode**: Prefix key `crysta1` did not match `crystal` under strict prefix equality.
- **Solution View**: Multi-token name pairs and number-street address matching anchor the candidate pair, allowing RapidFuzz character kernels to score the pair high in the matching model.

### Category 4: Token Reordering & Transposition
- **Diagnostic Observation**: Permutations in name tokens (e.g. `WordA WordB WordC` vs `WordA WordC WordB`).
- **Real Missed Example**:
  - **S1**: `Crystal Lending PC` | Target: `crystal pc lending`
- **Failure Mode**: Ordered prefix keys failed when middle tokens transposed.
- **Solution View**: Two-token symmetric pairs: `(tok0, tok1)` and `(tok1, tok0)` added to the inverted index.

### Category 5: Missing Building Numbers with Shared Street/City
- **Diagnostic Observation**: Target record omits the street number entirely, but retains the street name and city.
- **Real Missed Example**:
  - **S1**: `Christ Chapel` | Address: `2100 Cameron Drive, Dundalk, MD`
  - **Target (S2-103437512)**: `christ chapel` | Address: `cameron drive dundalk md`
- **Failure Mode**: Address number blocking failed because no digit existed in target.
- **Solution View**: Multi-view exact name index recovered this pair directly via `name_compact`.
