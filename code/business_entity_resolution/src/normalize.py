"""
Text normalization utilities for the Business Entity Resolution challenge.

All matching is done on *normalized* strings so that the many surface-form
variations in the data (case, diacritics, legal suffixes, street-type
abbreviations, US/India/France state names, transliterated Indic scripts,
punctuation) collapse to a common form.

Nothing here performs any external data lookup: every mapping is a small,
self-contained lexical table (suffix expansions, street types, state names).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from unidecode import unidecode

# --------------------------------------------------------------------------
# Character-level normalization
# --------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    """Remove combining diacritics (é -> e)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


@lru_cache(maxsize=1_000_000)
def norm_text(s: str) -> str:
    """Lower-cased, transliterated, accent-free, punctuation-free text.

    Indic scripts are transliterated to (approximate) Latin so that Devanagari
    / Gujarati / Tamil address components at least become Latin tokens instead
    of being dropped by the alphanumeric filter.
    """
    if not s:
        return ""
    # Fast path: pure ASCII lower-case already.
    try:
        s.encode("ascii")
        out = s.lower()
    except UnicodeEncodeError:
        out = unidecode(strip_accents(s)).lower()
    out = _NON_ALNUM.sub(" ", out)
    return _WS.sub(" ", out).strip()


def is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


# --------------------------------------------------------------------------
# Lexical tables
# --------------------------------------------------------------------------

# Legal-form tokens -> canonical short form.  The two sides of every pair are
# collapsed together so "Corporation"/"Corp"/"CORP." all become "corp".
LEGAL_TOKENS = {
    "incorporated": "inc", "inc": "inc", "incorporation": "inc",
    "corporation": "corp", "corp": "corp", "corporations": "corp",
    "company": "co", "co": "co", "companies": "co",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "proprietary": "pty", "pty": "pty",
    "llc": "llc", "lllp": "lllp", "llp": "llp", "lp": "lp", "plc": "plc",
    "solutions": "solutions",
    "gmbh": "gmbh", "sarl": "sarl", "sa": "sa", "sas": "sas", "ag": "ag",
    "bv": "bv", "nv": "nv", "spa": "spa", "sl": "sl", "eurl": "eurl",
}

# Business-name noise prefixes / connectors that carry no discriminative signal.
NAME_NOISE_TOKENS = {
    "the", "a", "an", "of", "and", "or", "at", "in", "on", "for", "to",
    "m", "ms", "mr", "mrs", "miss", "dr", "sri", "shree", "shri", "smt",
    "messrs", "aka", "dba", "fka", "trading", "as", "t",
}

# Street-type tokens -> canonical short form.
STREET_TYPES = {
    "street": "st", "st": "st", "str": "st", "saint": "st",
    "road": "rd", "rd": "rd", "route": "rte", "rte": "rte",
    "drive": "dr", "dr": "dr", "doctor": "dr",
    "lane": "ln", "ln": "ln",
    "avenue": "ave", "ave": "ave", "av": "ave", "avenida": "ave",
    "boulevard": "blvd", "blvd": "blvd", "boul": "blvd", "blv": "blvd",
    "court": "ct", "ct": "ct",
    "circle": "cir", "cir": "cir",
    "place": "pl", "pl": "pl", "plaza": "plz", "plz": "plz",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "square": "sq", "sq": "sq",
    "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl", "tr": "trl",
    "way": "way", "walk": "walk", "row": "row",
    "cross": "cross", "main": "main", "marg": "marg", "nagar": "nagar",
    "colony": "colony", "sector": "sector", "block": "blk", "blk": "blk",
    "plot": "plot", "survey": "surv", "surv": "surv",
    "chemin": "chemin", "impasse": "impasse", "allee": "allee",
    "quai": "quai", "passage": "passage", "voie": "voie",
}

# Address tokens that are pure boilerplate / mailing artefacts.
ADDR_NOISE_TOKENS = {
    "po", "pob", "box", "pmb", "po box", "pobox", "cmb",
    "unit", "apt", "apartment", "suite", "ste", "floor", "flr", "fl",
    "building", "bldg", "shop", "flat", "room", "rm", "no", "num", "number",
    "hn", "house", "hno", "dno", "door", "near", "opp", "opposite", "behind",
    "beside", "adjacent", "off", "above", "below", "next", "landmark",
    "dist", "district", "post", "taluka", "tehsil", "village", "vill",
    "pin", "pincode", "zip", "zipcode", "gpo",
}

# Strongly-shared, low-signal tokens (used for IDF down-weighting / pruning).
GENERIC_TOKENS = (
    set(LEGAL_TOKENS) | set(STREET_TYPES) | set(ADDR_NOISE_TOKENS)
    | set(NAME_NOISE_TOKENS)
)

# State / province names -> canonical 2-letter-ish code.  Used only to make
# equivalent spellings comparable ("Ohio" == "OH", "Maharashtra" == "MH"); a
# token that is not in the map is left untouched, so unseen countries (e.g.
# France) still work.
_STATE_TABLE = {
    # --- United States ---
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct",
    "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr", "virgin islands": "vi", "guam": "gu",
    # --- India (English names) ---
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj",
    "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl",
    "new delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la",
    "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
    # --- France (regions / common) ---
    "ile de france": "idf", "nouvelle aquitaine": "naq",
    "auvergne rhone alpes": "ara", "occitanie": "occ",
    "provence alpes cote d azur": "paca", "bretagne": "bre",
    "normandie": "nor", "pays de la loire": "pdl", "grand est": "ges",
    "hauts de france": "hdf", "bourgogne franche comte": "bfc",
    "centre val de loire": "cvl", "corse": "cor",
    # US/India two-letter codes are themselves valid tokens; keep identity.
}
STATE_TOKENS = {}
for _k, _v in _STATE_TABLE.items():
    STATE_TOKENS[_k] = _v
    for _t in _k.split():
        STATE_TOKENS.setdefault(_t, _v)
# Identity entries for short codes.
for _v in set(_STATE_TABLE.values()):
    STATE_TOKENS.setdefault(_v, _v)

# Region-name variants with accents / apostrophes already stripped by norm_text.


# --------------------------------------------------------------------------
# Tokenization helpers
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(s: str) -> list[str]:
    """Normalized whitespace-delimited tokens of ``s``."""
    return norm_text(s).split()


def name_tokens(name: str) -> list[str]:
    """Canonicalized name tokens (legal forms collapsed, noise words dropped).

    Multi-word state/legal phrases are collapsed first (``new york`` -> ``ny``).
    """
    toks = _canon_multiword(tokens(name))
    out = []
    for t in toks:
        t = LEGAL_TOKENS.get(t, t)
        if t in NAME_NOISE_TOKENS:
            continue
        out.append(t)
    return out


def addr_tokens(addr: str) -> list[str]:
    """Canonicalized address tokens (street types + states canonicalized)."""
    toks = _canon_multiword(tokens(addr))
    out = []
    for t in toks:
        t = STREET_TYPES.get(t, t)
        t = STATE_TOKENS.get(t, t)
        out.append(t)
    return out


def _canon_multiword(toks: list[str]) -> list[str]:
    """Collapse known multi-word phrases ("new york" -> "ny", "po box" -> "pobox").

    Only phrases whose *first* word is actually present are considered, so the
    common case (no multi-word phrase) costs a couple of set lookups.
    """
    if len(toks) < 2:
        return toks
    present = set(toks)
    hits = [w for w in present if w in _MULTIWORD_FIRST]
    if not hits:
        return toks
    s = " " + " ".join(toks) + " "
    for first in hits:
        for phrase, rep in _MULTIWORD_FIRST[first]:
            if phrase in s:
                s = s.replace(phrase, " " + rep + " ")
    return s.split()


_MULTIWORD = {
    "new york": "ny", "new jersey": "nj", "new mexico": "nm",
    "new hampshire": "nh", "new delhi": "dl", "north carolina": "nc",
    "south carolina": "sc", "north dakota": "nd", "south dakota": "sd",
    "west bengal": "wb", "west virginia": "wv", "rhode island": "ri",
    "po box": "pobox", "p o box": "pobox", "post office box": "pobox",
    "tamil nadu": "tn", "andhra pradesh": "ap", "madhya pradesh": "mp",
    "uttar pradesh": "up", "himachal pradesh": "hp", "arunachal pradesh": "ar",
    "jammu and kashmir": "jk", "andaman and nicobar islands": "an",
    "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "ile de france": "idf", "nouvelle aquitaine": "naq",
    "auvergne rhone alpes": "ara", "provence alpes cote d azur": "paca",
    "pays de la loire": "pdl", "grand est": "ges", "hauts de france": "hdf",
    "bourgogne franche comte": "bfc", "centre val de loire": "cvl",
    "district of columbia": "dc",
}

# Index the multi-word table by its first word for a fast lookup.
_MULTIWORD_FIRST: dict[str, list[tuple[str, str]]] = {}
for _phrase, _rep in _MULTIWORD.items():
    _MULTIWORD_FIRST.setdefault(_phrase.split()[0], []).append((_phrase, _rep))

_NUM_RE = re.compile(r"\d+")


def numeric_tokens(s: str) -> list[str]:
    """All digit runs in the normalized text."""
    return _NUM_RE.findall(norm_text(s))


def digit_signature(s: str) -> str:
    """Concatenated, order-preserving digits (used as a blocking key)."""
    return "".join(_NUM_RE.findall(norm_text(s)))


# --------------------------------------------------------------------------
# Similarity primitives (pure-python; heavy lifting uses rapidfuzz elsewhere)
# --------------------------------------------------------------------------

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def overlap_coef(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))
