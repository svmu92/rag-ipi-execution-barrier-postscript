"""Assembles the adversarial passage for any (query, condition, rho) -- plan
section 6, Table "ID / Mechanism / Obfuscated span / What it tests".

Layout is always: trigger, then instruction, then canary in clear, last
(plan section 6, "How rho is defined").

Decision D1 (confirmed by the user, 2026-09-19): in C1-C5 the trigger and
the instruction are obfuscated *together*, so retrievability bears the full
cost of obfuscation. C6 is the hybrid from Chang et al. [7]: the trigger
stays clean and only the instruction is obfuscated, under whichever
mechanism the week-2 pilot (rung R5) selects as M*.
"""
from __future__ import annotations

from dataclasses import dataclass

from .obfuscate import MECHANISMS, obfuscate_span
from .payload import instruction_for, trigger_known_query

CONDITIONS = ("C1_tags", "C2_zerowidth", "C3_homoglyph", "C4_typoglycemia", "C5_base64", "C6_hybrid")

MECHANISM_BY_CONDITION = {
    "C1_tags": "tags",
    "C2_zerowidth": "zerowidth",
    "C3_homoglyph": "homoglyph",
    "C4_typoglycemia": "typoglycemia",
    "C5_base64": "base64",
    # C6_hybrid has no fixed mechanism: pass hybrid_mechanism explicitly.
}


@dataclass
class AdversarialPassage:
    text: str
    query_id: str
    condition: str
    rho: float
    canary: str
    trigger_clean: bool  # True only for C6_hybrid


def build_payload_only(query_id: str, seed: int = 0) -> AdversarialPassage:
    """Rung R2 control: the instruction with no retrieval trigger at all --
    the retrieval floor an unoptimised injection starts from."""
    instruction, canary = instruction_for(query_id, seed=seed)
    text = f"{instruction} {canary}"
    return AdversarialPassage(text, query_id, "payload_only", 0.0, canary, trigger_clean=False)


def build_trigger_plus_payload(query_text: str, query_id: str, seed: int = 0) -> AdversarialPassage:
    """Rung R2 treatment: clean trigger + clean instruction (rho=0 by
    construction for every mechanism, since nothing is obfuscated yet)."""
    instruction, canary = instruction_for(query_id, seed=seed)
    trigger = trigger_known_query(query_text)
    text = f"{trigger} {instruction} {canary}"
    return AdversarialPassage(text, query_id, "trigger_plus_payload", 0.0, canary, trigger_clean=True)


def build_adversarial(
    query_text: str,
    query_id: str,
    condition: str,
    rho: float,
    seed: int = 0,
    hybrid_mechanism: str | None = None,
) -> AdversarialPassage:
    """Build the adversarial passage for one (query, condition, rho).

    condition: one of CONDITIONS ('C1_tags' ... 'C6_hybrid').
    hybrid_mechanism: required for 'C6_hybrid', one of src.attack.obfuscate.MECHANISMS
        (the mechanism chosen in the week-2 pilot, rung R5).
    """
    instruction, canary = instruction_for(query_id, seed=seed)
    trigger = trigger_known_query(query_text)
    obf_seed = f"{seed}-{query_id}-{condition}"

    if condition == "C6_hybrid":
        if hybrid_mechanism is None:
            raise ValueError(
                "C6_hybrid requires hybrid_mechanism (one of src.attack.obfuscate.MECHANISMS); "
                "pick it from the week-2 pilot (rung R5) before running the main sweep."
            )
        if hybrid_mechanism not in MECHANISMS:
            raise ValueError(f"unknown hybrid_mechanism {hybrid_mechanism!r}")
        obf = obfuscate_span(instruction, hybrid_mechanism, rho, seed=obf_seed)
        text = f"{trigger} {obf.text} {canary}"
        return AdversarialPassage(text, query_id, condition, rho, canary, trigger_clean=True)

    if condition not in MECHANISM_BY_CONDITION:
        raise ValueError(f"unknown condition {condition!r}, expected one of {CONDITIONS}")
    mechanism = MECHANISM_BY_CONDITION[condition]
    span = f"{trigger} {instruction}"
    obf = obfuscate_span(span, mechanism, rho, seed=obf_seed)
    text = f"{obf.text} {canary}"
    return AdversarialPassage(text, query_id, condition, rho, canary, trigger_clean=False)


def build_all_conditions(
    query_text: str,
    query_id: str,
    rhos: list[float],
    seed: int = 0,
    hybrid_mechanism: str | None = None,
) -> list[AdversarialPassage]:
    """Convenience sweep: every (condition, rho) pair for one query, matching
    the factorial design in plan section 7 (25 configs per query: C1-C5 x
    4 non-zero rho + 1 shared rho=0 baseline, plus C6)."""
    out = []
    for condition in CONDITIONS:
        for rho in rhos:
            if condition == "C6_hybrid" and hybrid_mechanism is None:
                continue  # skip until M* is chosen (week 2)
            out.append(
                build_adversarial(
                    query_text, query_id, condition, rho, seed=seed, hybrid_mechanism=hybrid_mechanism
                )
            )
    return out
