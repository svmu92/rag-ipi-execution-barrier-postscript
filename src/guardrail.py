"""Guardrail wrapper: scores a passage's injection probability using an
open-source HF text-classification model (plan section 5 rung R4, section 7).

Two models are named in the plan: ProtectAI's DeBERTa v3 prompt-injection
classifier (main) and Meta's Llama Prompt Guard 2 86M (secondary, gated,
BENIGN/MALICIOUS labels confirmed against its model card). `load_guardrail`
needs `transformers` + a downloaded model + (ideally) a GPU; its import is
deferred so this module loads without them -- score_passages/epsilon_from_results
are plain Python and are exercised in tests/test_guardrail_generate.py with a
stub classifier, no transformers needed.

IMPORTANT: confirm the exact label strings each checkpoint returns during
the week-1 smoke test (plan section 4 step 6, `clf([...])`) before trusting
MALICIOUS_LABEL_GUESS below -- id2label naming varies by checkpoint.
ProtectAI's label strings are NOT independently verified in this repo.
"""
from __future__ import annotations

from dataclasses import dataclass

MALICIOUS_LABEL_GUESS = {
    # UNVERIFIED -- print one raw pipeline output in the week-1 smoke test
    # (README.md Step 6) and correct this if it differs.
    "protectai/deberta-v3-base-prompt-injection-v2": "INJECTION",
    # Confirmed against the model card during planning.
    "meta-llama/Llama-Prompt-Guard-2-86M": "MALICIOUS",
}


@dataclass
class GuardrailResult:
    text: str
    predicted_label: str
    predicted_score: float
    injection_probability: float | None  # None if malicious_label wasn't among the returned labels


def load_guardrail(model_name: str, device: int = 0):
    """Deferred import: needs transformers + the downloaded model (and
    ideally a GPU, though these classifiers are small enough for CPU too)."""
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=model_name,
        truncation=True,
        max_length=512,
        top_k=None,  # return every class's score, not just the argmax
        device=device,
    )


def score_passages(clf, texts: list[str], malicious_label: str, batch_size: int = 32) -> list[GuardrailResult]:
    """Score a batch of passages with an HF text-classification pipeline (or
    any duck-typed stand-in: callable as clf(texts, batch_size=...) ->
    list[list[{"label": str, "score": float}]], one inner list per text,
    covering every class). `malicious_label` must be the exact label string
    this checkpoint uses for the injection/malicious class."""
    raw = clf(texts, batch_size=batch_size)
    out = []
    for text, scores in zip(texts, raw):
        by_label = {s["label"]: s["score"] for s in scores}
        predicted = max(scores, key=lambda s: s["score"])
        out.append(
            GuardrailResult(
                text=text,
                predicted_label=predicted["label"],
                predicted_score=predicted["score"],
                injection_probability=by_label.get(malicious_label),
            )
        )
    return out


def epsilon_from_results(results: list[GuardrailResult], malicious_label: str) -> float:
    """epsilon = Pr[guardrail labels the passage benign] = 1 - Pr[predicted
    label == malicious] (plan eq. 2)."""
    if not results:
        return 0.0
    flagged = sum(1 for r in results if r.predicted_label == malicious_label)
    return 1 - flagged / len(results)
