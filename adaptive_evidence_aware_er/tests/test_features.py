import pytest
from src.features import extract_pair_features, FEATURE_NAMES
from src.normalization import normalize_business_name, normalize_business_address

def test_extract_pair_features_dimension():
    s1_n, s1_c, s1_t, _ = normalize_business_name("Maure Williams Colombier Inc")
    s1_an, _, s1_at, s1_anum = normalize_business_address("85 Wayne Ave, NY")
    
    cand_n, cand_c, cand_t, _ = normalize_business_name("Maure Wilblims Colombier")
    cand_an, _, cand_at, cand_anum = normalize_business_address("85 Wayne Avenue, NY")
    
    feats = extract_pair_features(
        s1_n, s1_c, s1_t, s1_an, s1_at, s1_anum,
        "S2-1234", cand_n, cand_c, cand_t, cand_an, cand_at, cand_anum,
        rule_hits=3
    )
    
    assert len(feats) == len(FEATURE_NAMES)
    assert feats[1] > 0.85  # high jaro-winkler
    assert feats[15] == 3.0 # rule hits
    assert feats[16] == 1.0 # is S2
