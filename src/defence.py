"""Normalisation defence layer (plan section 8): sanitise() variants V0-V5.

This layer can only see and remove *Unicode-level* carriers -- Tags-block
invisibility, zero-width insertion, and cross-script homoglyphs. It is blind
to typoglycemia and Base64, which stay plain ASCII throughout (hypothesis
H4 in the plan: "Unicode-level carriers are largely neutralised; ASCII-level
carriers are not touched by it" -- see test_defence.py for the check that
confirms this split, which is an expected result, not a bug).

Two important asymmetries to know before reading results:
  - Tags: sanitising DELETES the hidden word outright (there is no visible
    form to recover -- the characters it was built from are themselves the
    payload). V2's "reveal" mode converts them to visible ASCII first, so a
    downstream guardrail gets a chance to flag the recovered text instead of
    the payload silently vanishing.
  - Zero-width: sanitising recovers the original visible word exactly,
    because the visible characters were never replaced, only interleaved
    with U+200B.
  - Homoglyph (V3): a word in which *every* letter has a look-alike becomes
    pure non-Latin script and is invisible to the mixed-script check -- see
    test_defence.py::test_v3_misses_fully_substituted_word_v4_catches_it.
    V4 catches it, at the cost of also rewriting genuine Cyrillic/Greek text.
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import regex
from confusable_homoglyphs import confusables

from src.attack.obfuscate import HOMOGLYPH_REVERSE, TAGS_ASCII_HIGH, TAGS_ASCII_LOW, TAGS_BASE

LATIN_RE = regex.compile(r"\p{Script=Latin}")
BASE64_TOKEN_RE = re.compile(r"[A-Za-z0-9+/]+=*")

VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5")


@dataclass
class DefenceResult:
    text: str
    changed: bool


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def strip_format_chars(text: str) -> str:
    """NFKC-normalise, then delete every character of Unicode category Cf
    (zero-width joiners, bidi controls, the Tags block)."""
    t = unicodedata.normalize("NFKC", text)
    return "".join(c for c in t if unicodedata.category(c) != "Cf")


def reveal_tags(text: str) -> str:
    """V2's audit mode: convert Tags-block characters back to visible ASCII
    instead of deleting them, so a downstream guardrail can read the
    previously-hidden text rather than it silently disappearing."""
    out = []
    for c in text:
        cp = ord(c)
        if TAGS_BASE + TAGS_ASCII_LOW <= cp <= TAGS_BASE + TAGS_ASCII_HIGH:
            out.append(chr(cp - TAGS_BASE))
        else:
            out.append(c)
    return "".join(out)


@lru_cache(maxsize=8192)
def _latin_equivalent(char: str) -> str | None:
    """Best-effort Latin look-alike for a single non-Latin character: first
    our own attack's exact table (HOMOGLYPH_REVERSE, from src/attack/obfuscate.py),
    then a fallback query against the UTS #39 confusables data via the
    confusable-homoglyphs package, for coverage beyond our own attack set."""
    if char in HOMOGLYPH_REVERSE:
        return HOMOGLYPH_REVERSE[char]
    if not char.isalpha() or LATIN_RE.match(char):
        return None
    try:
        candidates = confusables.is_confusable(char, greedy=True) or []
    except Exception:
        return None
    for entry in candidates:
        for h in entry.get("homoglyphs", []):
            if len(h["c"]) == 1 and h["c"].isascii() and h["c"].isalpha():
                return h["c"]
    return None


def map_confusables(text: str, mixed_script_only: bool = True) -> str:
    """Map non-Latin look-alike letters back to Latin.

    mixed_script_only=True (V3): only touches words that mix Latin with
    another script -- leaves genuine non-Latin text alone, but misses a word
    where *every* letter was substituted (it then reads as pure non-Latin
    script; see module docstring).
    mixed_script_only=False (V4): maps every word, catching that case at the
    cost of rewriting real Cyrillic/Greek text too (the utility cost
    measured in plan section 9, Table 2).
    """
    out_words = []
    for word in text.split(" "):
        if mixed_script_only and not confusables.is_mixed_script(word):
            out_words.append(word)
            continue
        out_words.append("".join(_latin_equivalent(c) or c for c in word))
    return " ".join(out_words)


def detect_and_decode_base64(text: str, min_len: int = 8) -> str:
    """V5 (stretch, plan section 8 step 5): find base64-looking tokens and
    append a decoded hint beside them, so a downstream guardrail or
    generator sees the plaintext instead of only the opaque blob. This is an
    audit aid -- it does not remove the blob, so it does not by itself stop
    the generator from also decoding and following it."""
    out_tokens = []
    for token in text.split(" "):
        out_tokens.append(token)
        candidate = token.strip(".,:;!?\"'()[]{}")
        if len(candidate) >= min_len and BASE64_TOKEN_RE.fullmatch(candidate):
            padded = candidate + "=" * (-len(candidate) % 4)
            try:
                decoded = base64.b64decode(padded, validate=True).decode("utf-8")
            except (binascii.Error, ValueError, UnicodeDecodeError):
                continue
            if decoded.isprintable():
                out_tokens.append(f"[decoded: {decoded}]")
    return " ".join(out_tokens)


# --------------------------------------------------------------------------
# The variants table from plan section 8
# --------------------------------------------------------------------------

def sanitise(text: str, variant: str = "V1") -> DefenceResult:
    """Apply one defence variant.

    V0 none | V1 NFKC + strip Cf | V2 reveal Tags, then strip Cf |
    V3 V1 + confusable map (mixed-script words only) |
    V4 V3's map but on every word | V5 V3 + base64 decode-and-annotate.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    if variant == "V0":
        return DefenceResult(text, changed=False)

    t = reveal_tags(text) if variant == "V2" else text
    t = strip_format_chars(t)
    if variant in ("V3", "V4"):
        t = map_confusables(t, mixed_script_only=(variant == "V3"))
    if variant == "V5":
        t = map_confusables(t, mixed_script_only=True)
        t = detect_and_decode_base64(t)
    return DefenceResult(t, changed=(t != text))


def sanitise_batch(texts: list[str], variant: str = "V1") -> list[DefenceResult]:
    return [sanitise(t, variant) for t in texts]
