import pytest
from src.blocking import MultiViewBlockingIndex
from src.normalization import normalize_business_name, normalize_business_address

def test_blocking_exact_and_prefix():
    index = MultiViewBlockingIndex(country="US")
    
    # Add target record S2-001
    n_norm, n_comp, n_tok, _ = normalize_business_name("Maure Wilblims Colombier Inc")
    a_norm, _, a_tok, a_num = normalize_business_address("85 Wayne Ave, NY")
    index.add_target_record("S2-001", n_norm, n_comp, n_tok, a_norm, a_tok, a_num)
    
    # Query with S1-001
    q_norm, q_comp, q_tok, _ = normalize_business_name("Maure Williams Colombier")
    q_anorm, _, q_atok, q_anum = normalize_business_address("85 Wayne Avenue, NY")
    
    candidates = index.query_s1_entity(q_norm, q_comp, q_tok, q_anorm, q_atok, q_anum)
    cand_ids = [c[0] for c in candidates]
    assert "S2-001" in cand_ids

def test_blocking_address_match_with_noisy_name():
    index = MultiViewBlockingIndex(country="India")
    
    # Target record with Tamil name but identical address
    n_norm, n_comp, n_tok, _ = normalize_business_name("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி")
    a_norm, _, a_tok, a_num = normalize_business_address("6(29), C.I.T. Colony, 2nd Main Road Mylapore Chennai")
    index.add_target_record("S2-TAMIL", n_norm, n_comp, n_tok, a_norm, a_tok, a_num)
    
    # Query with English S1
    q_norm, q_comp, q_tok, _ = normalize_business_name("Raj Investments LLP")
    q_anorm, _, q_atok, q_anum = normalize_business_address("6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu")
    
    candidates = index.query_s1_entity(q_norm, q_comp, q_tok, q_anorm, q_atok, q_anum)
    cand_ids = [c[0] for c in candidates]
    assert "S2-TAMIL" in cand_ids
