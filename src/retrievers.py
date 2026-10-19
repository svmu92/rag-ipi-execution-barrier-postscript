"""Retriever wrappers for BM25, E5-base-v2 and Contriever-msmarco (plan
section 4 step 4, section 5 rungs R1-R2, section 7).

BM25 runs anywhere (bm25s is pure Python/numpy) and is fully exercised by
tests/test_retrievers.py in this repo, with no GPU needed. E5 and Contriever
need `torch`, `sentence-transformers` / `transformers`, and a GPU for corpus-
scale work; those imports are deferred into the functions that need them so
this module still imports cleanly on a machine without torch (e.g. this
development sandbox) -- only calling encode_corpus_dense/EmbeddingIndex
requires the GPU environment from README.md Step 2.

Implements the "outside-the-index" scoring shortcut from plan section 5,
rung R2: scoring one adversarial passage against the cached clean top-100
scores, without rebuilding the real index. For BM25 this file's external
scorer is provably identical to bm25s's own 'lucene' method (same idf and tf
formulas, verified against bm25s.scoring source -- see docs/BM25_NOTES.md),
and tests/test_retrievers.py checks agreement against a real bm25s index.
For the dense retrievers, validate the shortcut for real once per plan
section 5 ("physically insert 20 adversarial passages into the real index")
before trusting it for the full sweep -- see validate_against_real_index().
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import bm25s
import numpy as np

try:
    import Stemmer

    _STEMMER = Stemmer.Stemmer("english")
    DEFAULT_STEMMER = _STEMMER.stemWords
except ImportError:  # pragma: no cover - PyStemmer is in requirements.txt
    DEFAULT_STEMMER = None

BM25_K1 = 1.5
BM25_B = 0.75


# ==========================================================================
# BM25 (sparse) -- fully runnable on CPU, no GPU needed
# ==========================================================================

def tokenize(texts: list[str], stopwords: str = "en", stemmer=DEFAULT_STEMMER) -> list[list[str]]:
    """Tokenize with the exact settings used for both the real index and the
    external scorer below, so ranks are comparable (plan section 5, R2)."""
    return bm25s.tokenize(
        texts, stopwords=stopwords, stemmer=stemmer, return_ids=False, show_progress=False
    )


@dataclass
class BM25Index:
    retriever: "bm25s.BM25"
    doc_ids: list[str]
    corpus_stats: "BM25Stats"


@dataclass
class BM25Stats:
    """Frozen corpus statistics needed to score a document that is NOT in the
    index, without rebuilding it. N = corpus size, avgdl = mean document
    length in tokens, df = term -> number of documents containing it."""

    n_docs: int
    avg_doc_len: float
    df: Counter = field(default_factory=Counter)


def _compute_stats(tokenized_corpus: list[list[str]]) -> BM25Stats:
    df = Counter()
    total_len = 0
    for doc_tokens in tokenized_corpus:
        total_len += len(doc_tokens)
        df.update(set(doc_tokens))
    n_docs = len(tokenized_corpus)
    return BM25Stats(n_docs=n_docs, avg_doc_len=total_len / max(n_docs, 1), df=df)


def build_bm25_index(doc_ids: list[str], texts: list[str], stopwords: str = "en") -> BM25Index:
    tokenized = tokenize(texts, stopwords=stopwords)
    retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B, method="lucene")
    retriever.index(tokenized, show_progress=False)
    stats = _compute_stats(tokenized)
    return BM25Index(retriever=retriever, doc_ids=doc_ids, corpus_stats=stats)


def bm25_top_k(index: BM25Index, query: str, k: int = 100, stopwords: str = "en") -> list[tuple[str, float]]:
    q_tokens = tokenize([query], stopwords=stopwords)
    results, scores = index.retriever.retrieve(q_tokens, k=min(k, len(index.doc_ids)), show_progress=False)
    return [(index.doc_ids[i], float(s)) for i, s in zip(results[0], scores[0])]


def bm25_score_external(query_tokens: list[str], doc_tokens: list[str], stats: BM25Stats,
                         k1: float = BM25_K1, b: float = BM25_B) -> float:
    """Score a document that is NOT in the index, using the index's frozen
    (N, avgdl, df) -- bm25s 'lucene' method's exact idf and tf formulas, so
    this agrees with a real index rebuild (validated in test_retrievers.py)."""
    tf = Counter(doc_tokens)
    l_d = len(doc_tokens)
    l_avg = stats.avg_doc_len or 1.0
    score = 0.0
    for term in set(query_tokens):
        if term not in tf:
            continue
        df = stats.df.get(term, 0)
        idf = math.log(1 + (stats.n_docs - df + 0.5) / (df + 0.5))
        tf_component = tf[term] / (k1 * ((1 - b) + b * l_d / l_avg) + tf[term])
        score += idf * tf_component
    return score


def bm25_rank_external(query: str, doc_text: str, clean_top_scores: list[float],
                        stats: BM25Stats, stopwords: str = "en") -> tuple[int, float]:
    """Rank of `doc_text` for `query`, estimated outside the index: 1 + the
    number of the cached clean top-N scores that beat it. Returns (rank,
    score); rank = len(clean_top_scores) + 1 means "outside the cached top-N"
    (plan section 5, rung R2's adv_rank_dense sketch, BM25 version)."""
    q_tokens = tokenize([query], stopwords=stopwords)[0]
    d_tokens = tokenize([doc_text], stopwords=stopwords)[0]
    score = bm25_score_external(q_tokens, d_tokens, stats)
    rank = 1 + sum(1 for s in clean_top_scores if s > score)
    return rank, score


# ==========================================================================
# Dense retrievers (E5-base-v2, Contriever-msmarco) -- need torch + a GPU
# ==========================================================================

@dataclass
class DenseIndex:
    """doc_ids aligned to the rows of `embeddings` (float32, L2-normalised
    for E5's cosine similarity; raw for Contriever's dot product)."""

    doc_ids: list[str]
    embeddings: "np.ndarray"
    model_name: str
    similarity: str  # "cosine" | "dot"


@lru_cache(maxsize=4)
def _load_dense_model(model_name: str, device: str = "cuda"):
    """Deferred import: only needed on the GPU machine. See README.md Step 4
    for the exact model ids (intfloat/e5-base-v2, facebook/contriever-msmarco).
    Cached: encode_dense is typically called once per query in a loop over
    hundreds of queries, and reloading the model from disk each time would
    dominate runtime -- this ensures the weights are loaded once per
    (model_name, device) and reused."""
    import torch  # noqa: F401
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_dense(texts: list[str], model_name: str, kind: str, device: str = "cuda",
                  batch_size: int = 64, max_length: int = 512) -> "np.ndarray":
    """Encode `texts` with E5 or Contriever. `kind` is 'query' or 'passage':
    E5 requires the 'query: '/'passage: ' prefix (its model card); Contriever
    uses no prefix. Returns float32 embeddings, L2-normalised for E5 (so dot
    product == cosine similarity) and raw for Contriever (dot product only,
    per facebook/contriever-msmarco's usage)."""
    import torch
    import torch.nn.functional as F

    is_e5 = "e5" in model_name.lower()
    if is_e5:
        prefix = "query: " if kind == "query" else "passage: "
        texts = [prefix + t for t in texts]

    tokenizer, model = _load_dense_model(model_name, device=device)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            pooled = _mean_pool(hidden, enc["attention_mask"])
            if is_e5:
                pooled = F.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().float().numpy())
    return np.concatenate(out, axis=0)


def build_dense_index(doc_ids: list[str], texts: list[str], model_name: str,
                       device: str = "cuda", batch_size: int = 64) -> DenseIndex:
    embeddings = encode_dense(texts, model_name, kind="passage", device=device, batch_size=batch_size)
    similarity = "cosine" if "e5" in model_name.lower() else "dot"
    return DenseIndex(doc_ids=doc_ids, embeddings=embeddings, model_name=model_name, similarity=similarity)


def dense_top_k(index: DenseIndex, query: str, k: int = 100, device: str = "cuda") -> list[tuple[str, float]]:
    q_emb = encode_dense([query], index.model_name, kind="query", device=device)[0]
    scores = index.embeddings @ q_emb
    top = np.argsort(-scores)[:k]
    return [(index.doc_ids[i], float(scores[i])) for i in top]


def dense_rank_external(query: str, doc_text: str, clean_top_scores: list[float],
                         model_name: str, device: str = "cuda") -> tuple[int, float]:
    """Rank of `doc_text` for `query`, estimated outside the index -- encode
    both, score, and compare against the cached clean top-N scores (plan
    section 5, rung R2's adv_rank_dense sketch)."""
    q_emb = encode_dense([query], model_name, kind="query", device=device)[0]
    d_emb = encode_dense([doc_text], model_name, kind="passage", device=device)[0]
    score = float(q_emb @ d_emb)
    rank = 1 + sum(1 for s in clean_top_scores if s > score)
    return rank, score


def validate_against_real_index(doc_ids: Iterable[str], texts: Iterable[str], adv_text: str,
                                 query: str, k: int = 100, backend: str = "bm25", **kwargs):
    """Rebuild a real index including `adv_text` and compare its rank there to
    the outside-the-index estimate -- the one-time check the plan requires
    before trusting the shortcut for the full sweep (section 5, rung R2):
    'physically insert 20 adversarial passages into the real index and
    confirm the ranks match'. backend: 'bm25' (runs here) or 'dense' (needs
    the GPU model named in kwargs['model_name'])."""
    doc_ids = list(doc_ids) + ["__adv__"]
    texts = list(texts) + [adv_text]

    if backend == "bm25":
        idx = build_bm25_index(doc_ids, texts)
        ranked = bm25_top_k(idx, query, k=len(doc_ids))
        real_rank = next(i for i, (did, _) in enumerate(ranked, start=1) if did == "__adv__")

        clean_idx = build_bm25_index(doc_ids[:-1], texts[:-1])
        clean_scores = [s for _, s in bm25_top_k(clean_idx, query, k=k)]
        est_rank, _ = bm25_rank_external(query, adv_text, clean_scores, clean_idx.corpus_stats)
        return real_rank, est_rank

    model_name = kwargs["model_name"]
    device = kwargs.get("device", "cuda")
    full_idx = build_dense_index(doc_ids, texts, model_name, device=device)
    ranked = dense_top_k(full_idx, query, k=len(doc_ids), device=device)
    real_rank = next(i for i, (did, _) in enumerate(ranked, start=1) if did == "__adv__")

    clean_idx = build_dense_index(doc_ids[:-1], texts[:-1], model_name, device=device)
    clean_scores = [s for _, s in dense_top_k(clean_idx, query, k=k, device=device)]
    est_rank, _ = dense_rank_external(query, adv_text, clean_scores, model_name, device=device)
    return real_rank, est_rank


def recall_at_k(ranks: list[int], k_values: tuple[int, ...] = (1, 5, 10)) -> dict[int, float]:
    """Recall@k over a list of per-query ranks (plan section 5, 9): the
    fraction of queries whose adversarial passage ranked at or above k."""
    n = len(ranks)
    return {k: sum(1 for r in ranks if r <= k) / n for k in k_values} if n else {k: 0.0 for k in k_values}
