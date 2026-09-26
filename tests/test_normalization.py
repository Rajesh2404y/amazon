import pytest
from src.normalization import normalize_business_name, normalize_business_address

def test_name_normalization_legal_suffixes():
    norm, compact, tokens, ngrams = normalize_business_name("Maure Williams Colombier Inc.")
    assert "inc" not in tokens
    assert "maure" in tokens
    assert "williams" in tokens
    assert compact.startswith("maurewilliam")

def test_name_normalization_url():
    norm, compact, tokens, ngrams = normalize_business_name("maurewilliamscolombier.com")
    assert compact == "maurewilliamscolombier"

def test_name_normalization_ampersand():
    norm, compact, tokens, ngrams = normalize_business_name("Barnes & Noble Booksellers")
    assert "and" in tokens
    assert "noble" in tokens

def test_address_normalization_abbreviations():
    norm, compact, tokens, numbers = normalize_business_address("3315 FREMONT ST, PEORIA, IL")
    assert "street" in tokens
    assert "illinois" in tokens
    assert "3315" in numbers

def test_address_normalization_empty():
    norm, compact, tokens, numbers = normalize_business_address(None)
    assert norm == ""
    assert compact == ""
    assert len(tokens) == 0
    assert len(numbers) == 0

def test_address_normalization_null_string():
    norm, compact, tokens, numbers = normalize_business_address("null")
    assert norm == ""
    assert len(tokens) == 0
