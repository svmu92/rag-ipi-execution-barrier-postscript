"""BEIR data loading and seeded query sampling (plan section 4 step 5,
section 5 "Query sets").

`load_beir` needs the `beir` package and network access to download the
dataset zip, so its import is deferred -- this module loads fine without
`beir` installed. The seeded sampling and NQ-Open join logic are plain
Python, independent of beir, and are exercised with a small synthetic
BEIR-shaped dataset in tests/test_data.py.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def beir_url(dataset_name: str) -> str:
    return BEIR_BASE_URL.format(name=dataset_name)


def load_beir(dataset_name: str, out_dir: str = "data", split: str = "test"):
    """Download (if needed) and load a BEIR dataset (README.md Step 5).
    Returns (corpus, queries, qrels) exactly as beir's GenericDataLoader does:
      corpus:  {doc_id: {"title": str, "text": str}}
      queries: {query_id: question_text}
      qrels:   {query_id: {doc_id: relevance}}
    Needs network access and the `beir` package -- run on the lab machine,
    not in a sandbox with restricted egress.
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    path = util.download_and_unzip(beir_url(dataset_name), out_dir)
    return GenericDataLoader(data_folder=path).load(split=split)


def passage_text(doc: dict) -> str:
    """BEIR passage text used for indexing: title + text (README.md Step 4/5;
    plan section 4 step 3's repo layout, src/data.py)."""
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


@dataclass
class QuerySplit:
    evaluation_ids: list[str]
    development_ids: list[str]


def sample_queries(query_ids: list[str], n_eval: int = 500, n_dev: int = 100, seed: int = 0) -> QuerySplit:
    """Disjoint evaluation/development query-id samples with a fixed seed
    (plan section 5, "Query sets"). Tune wording and thresholds on
    development only; evaluate once on evaluation, never the reverse."""
    ids = sorted(query_ids)  # sort first: makes the sample independent of
                             # the source dict's (unordered) iteration order
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    if n_eval + n_dev > len(shuffled):
        raise ValueError(
            f"requested {n_eval}+{n_dev}={n_eval + n_dev} queries but only {len(shuffled)} available"
        )
    development_ids = shuffled[:n_dev]
    evaluation_ids = shuffled[n_dev : n_dev + n_eval]
    return QuerySplit(evaluation_ids=evaluation_ids, development_ids=development_ids)


# --------------------------------------------------------------------------
# NQ-Open answer join (plan section 5, "Query sets": "BEIR NQ has no short
# answer strings ... Join the queries to NQ-Open by normalised question
# text ... check the match rate in week 1" -- this was flagged UNVERIFIED in
# the plan; check the match rate before relying on it for the utility metric).
# --------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_question(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace -- the join key
    between BEIR NQ questions and NQ-Open questions."""
    text = text.lower().translate(_PUNCT_TABLE)
    return " ".join(text.split())


def join_nq_open_answers(beir_queries: dict[str, str], nq_open: list[dict]) -> dict[str, list[str]]:
    """beir_queries: {query_id: question_text} from BEIR's GenericDataLoader.
    nq_open: records with "question" (str) and "answer" (list[str]) keys,
    e.g. loaded from the NQ-Open dataset on the Hugging Face Hub
    (google-research-datasets/nq_open) with `datasets.load_dataset`.

    Returns {query_id: [gold answers]} for every BEIR query whose normalised
    text matches exactly one NQ-Open question. Unmatched queries are simply
    absent from the result -- report len(result) / len(beir_queries) in week
    1 (the plan's match-rate check) and fall back to the NQ-Open questions
    directly for utility if it is low, keeping the BEIR qrels for retrieval."""
    nq_open_index: dict[str, list[str]] = {}
    for row in nq_open:
        key = normalize_question(row["question"])
        nq_open_index.setdefault(key, row["answer"])

    out: dict[str, list[str]] = {}
    for qid, text in beir_queries.items():
        key = normalize_question(text)
        if key in nq_open_index:
            out[qid] = nq_open_index[key]
    return out


def match_rate(beir_queries: dict[str, str], joined: dict[str, list[str]]) -> float:
    """The week-1 check the plan calls for before trusting the join."""
    return len(joined) / len(beir_queries) if beir_queries else 0.0
