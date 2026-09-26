# Analysis of Missed Ground-Truth Matches (15,000 S1 Entities)

## 1. Quantitative Summary
- **Total S1 Sample**: 15,000 entities
- **Total Ground-Truth Matches**: 52,027
- **Retrieved Ground-Truth Matches**: 44,729 (85.97%)
- **Missed Ground-Truth Matches**: 7,298 (14.03%)
- **Current Blocking Architecture**: Option A (Name Top-25, Address Top-15, Char Top-10, Max Budget: 50)

## 2. Root Cause Taxonomy & Distribution

| Category | Missed Count | % of Missed | Primary Root Cause | Proposed Recovery Signal |
| :--- | :--- | :--- | :--- | :--- |
| **name_quota_overflow** | 4,806 | 65.85% | Shared exact 7-prefix or bigram, but target ranked > 25 in name postings | Adaptive name quota for entities with high name frequency |
| **token_reorder_or_partial_name** | 1,817 | 24.90% | Shared significant tokens but different token order or prefix word | Significant / rare token inverted index across any token position |
| **address_key_or_quota_miss** | 670 | 9.18% | Shared street number or street name, but address key missed or ranked > 15 | Street number + 1st significant name token compound blocking key |
| **missing_address_record** | 5 | 0.07% | Address empty or severely truncated in S2/S3 record | Allocate full quota to name & token views when address is missing |

## 3. Representative Case Studies

### Category: `address_key_or_quota_miss`
**Example 1:**
- **S1 Name**: `b retail`
- **Target Name**: `b services`
- **S1 Address**: `1712 montebello avenue phoenix az`
- **Target Address**: `arizona phoenix 1712 montebello avenue`
- **Street Number Overlap**: `['1712']`

**Example 2:**
- **S1 Name**: `helios`
- **Target Name**: `hiros`
- **S1 Address**: `66 edgewood street bridgeport court`
- **Target Address**: `connecticut 66 edgewood street bridgeport`
- **Street Number Overlap**: `['66']`

**Example 3:**
- **S1 Name**: `helios`
- **Target Name**: `heli0s lp`
- **S1 Address**: `66 edgewood street bridgeport court`
- **Target Address**: `66 edgewood street bridgeport court`
- **Street Number Overlap**: `['66']`

**Example 4:**
- **S1 Name**: `cascade allied telecom`
- **Target Name**: `catelecom`
- **S1 Address**: `165 linden street unit 106 wellesley ma`
- **Target Address**: `65 linden street wellesley ma`
- **Street Number Overlap**: `[]`

**Example 5:**
- **S1 Name**: `corner hypnosis`
- **Target Name**: `brixlyra`
- **S1 Address**: `26 stone street unit 3 beverly ma`
- **Target Address**: `pmb 9599 ma beverly 0026 stone street`
- **Street Number Overlap**: `[]`

### Category: `name_quota_overflow`
**Example 1:**
- **S1 Name**: `christ chapel`
- **Target Name**: `christ chapel`
- **S1 Address**: `2100 cameron drive unit apartment g dundalk md`
- **Target Address**: `cameron drive dundalk md`

**Example 2:**
- **S1 Name**: `christ chapel`
- **Target Name**: `christ chapel`
- **S1 Address**: `2100 cameron drive unit apartment g dundalk md`
- **Target Address**: `apartment g dundalk 210 cameron drive maryland`

**Example 3:**
- **S1 Name**: `christ chapel`
- **Target Name**: `christ center`
- **S1 Address**: `2100 cameron drive unit apartment g dundalk md`
- **Target Address**: `210 cameron drive apartment g dundalk maryland`

**Example 4:**
- **S1 Name**: `christ chapel`
- **Target Name**: `christ chape1`
- **S1 Address**: `2100 cameron drive unit apartment g dundalk md`
- **Target Address**: `2100 cameron drive dundalk md`

**Example 5:**
- **S1 Name**: `christ chapel`
- **Target Name**: `christ chapel`
- **S1 Address**: `2100 cameron drive unit apartment g dundalk md`
- **Target Address**: `210 cameron drive dundalk maryland`

### Category: `token_reorder_or_partial_name`
**Example 1:**
- **S1 Name**: `helios`
- **Target Name**: `helios`
- **S1 Address**: `66 edgewood street bridgeport court`
- **Target Address**: `bridgeport 66 edgewood street connecticut`
- **Token Overlap**: `['helios']`

**Example 2:**
- **S1 Name**: `scott eagle`
- **Target Name**: `scott ene`
- **S1 Address**: `4828 hedges avenue kansas city missouri`
- **Target Address**: `hedges avenue kansas city missouri`
- **Token Overlap**: `['scott']`

**Example 3:**
- **S1 Name**: `scott eagle`
- **Target Name**: `the scott eagle`
- **S1 Address**: `4828 hedges avenue kansas city missouri`
- **Target Address**: `4830 hedges avenue kansas city missouri`
- **Token Overlap**: `['scott', 'eagle']`

**Example 4:**
- **S1 Name**: `schaefer michael and silver associates`
- **Target Name**: `silver michael and schaefer ass0ciates`
- **S1 Address**: `32876 circle drive millsboro de`
- **Target Address**: `32876 cidcle drive millsboro de`
- **Token Overlap**: `['schaefer', 'and', 'silver', 'michael']`

**Example 5:**
- **S1 Name**: `thomas calisa`
- **Target Name**: `th0mas calisa`
- **S1 Address**: `126 d street pittsfield me`
- **Target Address**: `126 d stret pittsfield maine`
- **Token Overlap**: `['calisa']`

### Category: `missing_address_record`
**Example 1:**
- **S1 Name**: `b biotechnologies`
- **Target Name**: `b services`
- **S1 Address**: `328 buckingham street unit 414 columbus ohio`
- **Target Address**: ``
- **Jaro-Winkler Similarity**: `0.656`

**Example 2:**
- **S1 Name**: `spano s eye care`
- **Target Name**: `span0 s eye`
- **S1 Address**: `6646 firenza place bldg 25425 dublin ohio`
- **Target Address**: ``
- **Jaro-Winkler Similarity**: `0.907`

**Example 3:**
- **S1 Name**: `maid book store`
- **Target Name**: `maid b0ok`
- **S1 Address**: `6000 powder wood lane arlington texas`
- **Target Address**: ``
- **Jaro-Winkler Similarity**: `0.884`

**Example 4:**
- **S1 Name**: `d harvard`
- **Target Name**: `d harnord`
- **S1 Address**: `573 washington street boston ma`
- **Target Address**: ``
- **Jaro-Winkler Similarity**: `0.911`

**Example 5:**
- **S1 Name**: `granite p c`
- **Target Name**: `gradgte p c`
- **S1 Address**: `539 summer street weymouth ma`
- **Target Address**: ``
- **Jaro-Winkler Similarity**: `0.915`

