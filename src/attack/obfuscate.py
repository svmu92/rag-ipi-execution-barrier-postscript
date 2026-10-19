"""Obfuscation mechanisms for adversarial RAG passages.

Implements the five mechanisms from the literature review (Boucher et al. [8]
canonical taxonomy of imperceptible perturbations) plus the rho-intensity
control described in the research plan, section 6:

    C1 tags          -- Unicode Tags block (U+E0000), category Cf, invisible
    C2 zerowidth      -- U+200B inserted between every character of a word
    C3 homoglyph      -- Latin -> visually-identical Cyrillic/Greek codepoints
    C4 typoglycemia   -- interior letters scrambled, first/last letter fixed
    C5 base64         -- a contiguous tail of the span replaced by one b64 blob

Every mechanism is deterministic given (seed, rho), and selections are
*nested*: the set of words changed at rho=0.5 is a subset of the set changed
at rho=0.75, for the same seed. This removes selection noise when comparing
across the rho sweep (plan section 6, "How rho is defined").

Verified against tests/test_obfuscate.py (the 8 tests listed in plan section 6).
"""
from __future__ import annotations

import base64
import random
import re
import unicodedata
from dataclasses import dataclass, field

MECHANISMS = ("tags", "zerowidth", "homoglyph", "typoglycemia", "base64")

# --- Unicode Tags block: mirrors printable ASCII into U+E0000-U+E007E. ------
# Confirmed category Cf for the whole range (checked empirically; see README).
TAGS_BASE = 0xE0000
TAGS_ASCII_LOW, TAGS_ASCII_HIGH = 0x20, 0x7E

# --- Zero-width: inserted between every character of the chosen word. ------
ZWSP = "​"

# --- Homoglyphs: Latin -> visually-confusable Cyrillic/Greek codepoints. ---
# Deliberately small and exact (UTS #39 confusables), not exhaustive: every
# entry round-trips cleanly via HOMOGLYPH_REVERSE, which the defence layer's
# confusable-mapping step (src/defence.py) and tests/test_obfuscate.py both
# depend on. Extend with care -- an asymmetric entry breaks the round-trip test.
HOMOGLYPH_MAP = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р",
    "x": "х", "y": "у", "i": "і", "s": "ѕ", "j": "ј",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "I": "І", "K": "К", "M": "М", "O": "О", "P": "Р",
    "S": "Ѕ", "T": "Т", "X": "Х", "Y": "У",
}
HOMOGLYPH_REVERSE = {v: k for k, v in HOMOGLYPH_MAP.items()}

BASE64_HINT = "Decode this base64 and follow it:"

_LEAD_CORE_TRAIL_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE | re.DOTALL)


# --------------------------------------------------------------------------
# Low-level, per-string transforms (each with an inverse, used by tests and
# by the defence layer's "reveal" audit mode).
# --------------------------------------------------------------------------

def to_tags(s: str) -> str:
    """Mirror printable ASCII characters into the invisible Unicode Tags block."""
    return "".join(
        chr(TAGS_BASE + ord(c)) if TAGS_ASCII_LOW <= ord(c) <= TAGS_ASCII_HIGH else c
        for c in s
    )


def from_tags(s: str) -> str:
    """Inverse of to_tags. Any non-tag characters pass through unchanged."""
    out = []
    for c in s:
        cp = ord(c)
        if TAGS_BASE + TAGS_ASCII_LOW <= cp <= TAGS_BASE + TAGS_ASCII_HIGH:
            out.append(chr(cp - TAGS_BASE))
        else:
            out.append(c)
    return "".join(out)


def is_tag_char(c: str) -> bool:
    cp = ord(c)
    return (
        unicodedata.category(c) == "Cf"
        and TAGS_BASE + TAGS_ASCII_LOW <= cp <= TAGS_BASE + TAGS_ASCII_HIGH
    )


def to_zerowidth(s: str) -> str:
    return ZWSP.join(s)


def from_zerowidth(s: str) -> str:
    return s.replace(ZWSP, "")


def to_homoglyph(s: str) -> str:
    return "".join(HOMOGLYPH_MAP.get(c, c) for c in s)


def from_homoglyph(s: str) -> str:
    return "".join(HOMOGLYPH_REVERSE.get(c, c) for c in s)


def to_typoglycemia(word: str, seed: str) -> str:
    """Scramble interior letters, first and last letter fixed. Retries the
    shuffle until it differs from the original when that is possible; words
    with fewer than 2 distinct interior letters are returned unchanged
    (no scramble can differ from the original)."""
    if len(word) < 4:
        return word
    first, mid, last = word[0], list(word[1:-1]), word[-1]
    if len(set(mid)) <= 1:
        return word
    rng = random.Random(seed)
    original = tuple(mid)
    for _ in range(20):
        rng.shuffle(mid)
        if tuple(mid) != original:
            break
    return first + "".join(mid) + last


def to_base64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def from_base64(s: str) -> str:
    return base64.b64decode(s.encode("ascii")).decode("utf-8")


# --------------------------------------------------------------------------
# Word splitting and nested rho selection.
# --------------------------------------------------------------------------

def split_word(token: str) -> tuple[str, str, str]:
    """Split a whitespace-delimited token into (leading punctuation, core,
    trailing punctuation), e.g. '"Paris,' -> ('"', 'Paris', ',')."""
    m = _LEAD_CORE_TRAIL_RE.match(token)
    return m.group(1), m.group(2), m.group(3)


def select_indices(seed: str, rho: float, eligible: list[int]) -> list[int]:
    """Nested, seeded selection over `eligible` indices: the list returned for
    a smaller rho is always a subset of the list returned for a larger rho,
    given the same seed and eligible set (test 3 in test_obfuscate.py)."""
    order = list(eligible)
    random.Random(seed).shuffle(order)
    k = round(rho * len(order))
    return sorted(order[:k])


@dataclass
class ObfuscationResult:
    text: str
    changed_word_indices: list[int] = field(default_factory=list)


def obfuscate_span(text: str, mechanism: str, rho: float, seed: str) -> ObfuscationResult:
    """Obfuscate `text` (a trigger, instruction, or trigger+instruction span)
    under `mechanism` at intensity rho in [0, 1]. Deterministic given seed.

    rho is the fraction of *eligible* words transformed (typoglycemia only
    counts words of 4+ letters; base64 counts the fraction of the whole span,
    by word count, folded into one trailing blob -- see plan section 6).
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}, expected one of {MECHANISMS}")
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    tokens = text.split(" ")

    if mechanism == "base64":
        n = round(rho * len(tokens))
        if n == 0:
            return ObfuscationResult(text, [])
        head, tail = tokens[:-n], tokens[-n:]
        blob = to_base64(" ".join(tail))
        out_tokens = (head + [BASE64_HINT, blob]) if head else [BASE64_HINT, blob]
        return ObfuscationResult(" ".join(out_tokens), list(range(len(tokens) - n, len(tokens))))

    parts = [split_word(t) for t in tokens]
    eligible = [
        i for i, (_, core, _) in enumerate(parts)
        if core and (mechanism != "typoglycemia" or len(core) >= 4)
    ]
    chosen = set(select_indices(seed, rho, eligible))

    out_tokens = list(tokens)
    for i in chosen:
        lead, core, trail = parts[i]
        if mechanism == "tags":
            core = to_tags(core)
        elif mechanism == "zerowidth":
            core = to_zerowidth(core)
        elif mechanism == "homoglyph":
            core = to_homoglyph(core)
        elif mechanism == "typoglycemia":
            core = to_typoglycemia(core, seed=f"{seed}-{i}")
        out_tokens[i] = lead + core + trail

    return ObfuscationResult(" ".join(out_tokens), sorted(chosen))
