import re
import unicodedata
from typing import Tuple, List, Set, Dict, Any, Optional

# Precompiled Regexes
RE_URL = re.compile(r"^(?:https?://)?(?:www\.)?([a-zA-Z0-9_\-]+)\.(?:com|org|net|in|co|io|fr|us|biz|info)(?:/.*)?$", re.IGNORECASE)
RE_AMP = re.compile(r"&")
RE_PUNCT = re.compile(r"[^\w\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_COMPACT = re.compile(r"[^a-z0-9]")

# Legal Suffixes Regex (single pass)
LEGAL_SUFFIXES_PATTERN = re.compile(
    r"\b(?:private\s+limited|pvt\s+ltd|pvt|ltd|limited|incorporated|inc|corporation|corp|limited\s+liability\s+company|llc|llp|sarl|sas|sa|eurl|gie|company|co|enterprises|holding|holdings)\b",
    re.IGNORECASE
)

# Combined Fast Abbreviation Dictionary (O(1) word lookup)
ABBR_MAP = {
    # Street abbreviations
    "rd": "road",
    "st": "street",
    "saint": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "ter": "terrace",
    "pkwy": "parkway",
    "hwy": "highway",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "no": "number",
    # US State abbreviations
    "ny": "new york",
    "ca": "california",
    "tx": "texas",
    "fl": "florida",
    "il": "illinois",
    "pa": "pennsylvania",
    "oh": "ohio",
    "ga": "georgia",
    "nc": "north carolina",
    "mi": "michigan",
    "nj": "new jersey",
    "va": "virginia",
    "wa": "washington",
    "mo": "missouri",
    # India State abbreviations
    "tn": "tamil nadu",
    "up": "uttar pradesh",
    "rj": "rajasthan",
    "ka": "karnataka",
    "mh": "maharashtra",
    "dl": "delhi",
    "wb": "west bengal",
    "ts": "telangana",
    "ap": "andhra pradesh",
    "gj": "gujarat",
    "kl": "kerala",
    "mp": "madhya pradesh",
    "pb": "punjab",
    "hr": "haryana"
}

def fast_unicode_normalize(text: str) -> str:
    """Fast unicode normalization with ASCII bypass for speed."""
    if not text:
        return ""
    if text.isascii():
        return text
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def normalize_business_name(name: Optional[str]) -> Tuple[str, str, Tuple[str, ...], Set[str]]:
    """
    Fast normalization of business name returning:
    (name_norm, name_compact, name_tokens, name_char_ngrams)
    """
    if not name or not isinstance(name, str):
        return "", "", (), set()

    raw = fast_unicode_normalize(name.strip().lower())

    # Check URL
    url_m = RE_URL.match(raw)
    if url_m:
        raw = url_m.group(1).replace("-", " ").replace("_", " ")

    # & -> and
    if "&" in raw:
        raw = RE_AMP.sub(" and ", raw)

    # Strip legal suffixes
    cleaned = LEGAL_SUFFIXES_PATTERN.sub(" ", raw)

    # Punctuation to space
    cleaned = RE_PUNCT.sub(" ", cleaned)
    norm = RE_WHITESPACE.sub(" ", cleaned).strip()

    if not norm:
        # Fallback if legal suffix stripping emptied name
        cleaned = RE_PUNCT.sub(" ", raw)
        norm = RE_WHITESPACE.sub(" ", cleaned).strip()

    compact = RE_COMPACT.sub("", norm)
    tokens = tuple(t for t in norm.split() if len(t) > 1)

    if len(compact) >= 3:
        char_ngrams = {compact[i:i+3] for i in range(min(len(compact)-2, 12))}
    else:
        char_ngrams = {compact} if compact else set()

    return norm, compact, tokens, char_ngrams

def normalize_business_address(address: Optional[str]) -> Tuple[str, str, Tuple[str, ...], Tuple[str, ...]]:
    """
    Fast normalization of business address using O(1) dictionary word substitution.
    Returns: (addr_norm, addr_compact, addr_tokens, addr_numbers)
    """
    if not address or not isinstance(address, str):
        return "", "", (), ()

    raw = fast_unicode_normalize(address.strip().lower())
    if raw in ("null", "none", "nan", "-"):
        return "", "", (), ()

    # Extract digits before punctuation removal
    numbers = tuple(re.findall(RE_NUMBERS, raw))

    # Punctuation to space
    cleaned = RE_PUNCT.sub(" ", raw)
    words = cleaned.split()

    # O(1) dictionary word normalization
    norm_words = []
    for w in words:
        mapped = ABBR_MAP.get(w, w)
        if mapped:
            norm_words.append(mapped)

    norm = " ".join(norm_words).strip()
    compact = RE_COMPACT.sub("", norm)
    tokens = tuple(w for w in norm_words if len(w) > 1)

    return norm, compact, tokens, numbers
