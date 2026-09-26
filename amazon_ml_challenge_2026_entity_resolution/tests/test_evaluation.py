import pytest
from src.evaluation import compute_entity_f05, evaluate_macro_f05

def test_evaluation_challenge_example():
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    p, r, f05 = compute_entity_f05(pred, true)
    
    assert round(p, 3) == 0.667
    assert round(r, 3) == 1.000
    assert round(f05, 3) == 0.714

def test_evaluation_singleton():
    # Correct singleton: true is empty, pred is empty -> f05 = 1.0
    p, r, f05 = compute_entity_f05(set(), set())
    assert f05 == 1.0
    
    # False positive on singleton: true is empty, pred has ID -> f05 = 0.0
    p, r, f05 = compute_entity_f05({"S2-999"}, set())
    assert f05 == 0.0

def test_macro_evaluation():
    gt = {
        "S1-1": {"S2-1"},
        "S1-2": set(),  # singleton
    }
    preds = {
        "S1-1": {"S2-1"}, # 1.0
        "S1-2": set(),    # 1.0
    }
    results = evaluate_macro_f05(preds, gt)
    assert results["macro_f05"] == 1.0
    assert results["singleton_accuracy"] == 1.0
