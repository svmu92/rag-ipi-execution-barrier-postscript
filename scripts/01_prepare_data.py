#!/usr/bin/env python3
"""Download BEIR, sample queries, and (best-effort) join NQ-Open answers
(plan section 4 step 5, section 5 "Query sets"). Needs network access and
the `beir` + `datasets` packages -- run on the lab machine.

Writes under <data_dir>/<dataset>/:
  queries_eval.json      {query_id: question_text}, 500 queries
  queries_dev.json       {query_id: question_text}, 100 queries
  nq_open_answers.json    {query_id: [gold answers]} (only for dataset=nq)
  match_rate.txt          the week-1 check the plan calls for

Usage: python scripts/01_prepare_data.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config

from src.data import join_nq_open_answers, load_beir, match_rate, sample_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    dataset = cfg["data"]["dataset"]
    split = cfg["data"]["split"]
    seed = cfg["project"]["seed"]
    out_dir = Path(cfg["project"]["data_dir"]) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading/loading BEIR {dataset} ({split} split) -- this can take a while the first time...")
    corpus, queries, qrels = load_beir(dataset, out_dir=cfg["project"]["data_dir"], split=split)
    print(f"Corpus: {len(corpus)} passages. Queries: {len(queries)}.")

    split_ids = sample_queries(
        list(queries.keys()),
        n_eval=cfg["data"]["n_eval_queries"],
        n_dev=cfg["data"]["n_dev_queries"],
        seed=seed,
    )
    eval_queries = {qid: queries[qid] for qid in split_ids.evaluation_ids}
    dev_queries = {qid: queries[qid] for qid in split_ids.development_ids}

    (out_dir / "queries_eval.json").write_text(json.dumps(eval_queries, ensure_ascii=True, indent=2))
    (out_dir / "queries_dev.json").write_text(json.dumps(dev_queries, ensure_ascii=True, indent=2))
    print(f"Wrote {len(eval_queries)} evaluation + {len(dev_queries)} development query ids.")

    if dataset == "nq":
        try:
            from datasets import load_dataset

            print("Loading NQ-Open for short-answer strings (google-research-datasets/nq_open)...")
            nq_open = load_dataset("google-research-datasets/nq_open", split="validation")
            nq_open_records = [{"question": r["question"], "answer": r["answer"]} for r in nq_open]
            all_beir_queries = {**eval_queries, **dev_queries}
            joined = join_nq_open_answers(all_beir_queries, nq_open_records)
            rate = match_rate(all_beir_queries, joined)
            (out_dir / "nq_open_answers.json").write_text(json.dumps(joined, ensure_ascii=True, indent=2))
            (out_dir / "match_rate.txt").write_text(
                f"{rate:.4f}\n\nWEEK-1 CHECK (plan section 5, 'Query sets'):\n"
                f"{len(joined)}/{len(all_beir_queries)} sampled queries matched an NQ-Open question.\n"
                "If this is low, use the NQ-Open questions directly for the utility metric\n"
                "(src.metrics.exact_match/f1_score) and keep the BEIR qrels for the retrieval\n"
                "sanity check (rung R1) only -- do not silently drop unmatched queries from EM/F1.\n"
            )
            print(f"NQ-Open match rate: {rate:.1%} ({len(joined)}/{len(all_beir_queries)}). See match_rate.txt.")
        except Exception as e:
            print(f"WARNING: could not load/join NQ-Open ({e}). "
                  "Utility (EM/F1) will need NQ-Open questions directly as a fallback -- see plan section 5.")

    print(f"Done. Data written under {out_dir}/")


if __name__ == "__main__":
    main()
