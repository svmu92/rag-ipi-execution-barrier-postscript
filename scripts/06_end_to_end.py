#!/usr/bin/env python3
"""The end-to-end check (plan section 7): retrieve for real, filter with the
guardrail, generate, and judge -- the measured ASR_e2e to compare against
the derived product P_ret x epsilon x P_gen from scripts/04_run_sweep.py
(src.metrics.asr_derived vs asr_measured; see scripts/07_analyze.py Table 1).

Runs on one retriever/guardrail/generator combination (defaults: BM25,
the first configured guardrail, the first configured generator) across every
(condition, rho), optionally under a defence variant.

Usage:
  python scripts/06_end_to_end.py --config configs/default.yaml
  python scripts/06_end_to_end.py --config configs/default.yaml --defence V1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import JsonlWriter, load_config

from src.attack.build_docs import build_adversarial
from src.data import load_beir, passage_text
from src.defence import sanitise
from src.generate import build_prompt, generate_batch, judge_batch, load_generator
from src.guardrail import load_guardrail, score_passages
from src.metrics import asr_measured
from src.retrievers import bm25_rank_external, bm25_top_k, build_bm25_index


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--defence", default="V0", help="defence variant to apply after retrieval (default V0 = none)")
    parser.add_argument("--k", type=int, default=None, help="override generation_k from config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg["project"]["seed"]
    k = args.k or cfg["attack"]["generation_k"]
    hybrid_mech = cfg["attack"].get("hybrid_mechanism")

    eval_queries = json.loads((Path(cfg["project"]["data_dir"]) / cfg["data"]["dataset"] / "queries_eval.json").read_text())
    corpus, _, _ = load_beir(cfg["data"]["dataset"], out_dir=cfg["project"]["data_dir"], split=cfg["data"]["split"])
    doc_ids = list(corpus.keys())
    texts = [passage_text(corpus[d]) for d in doc_ids]
    text_by_id = dict(zip(doc_ids, texts))
    bm25_index = build_bm25_index(doc_ids, texts)

    g_cfg = cfg["guardrails"][0]
    gen_cfg = cfg["generators"][0]
    print(f"End-to-end pipeline: BM25 -> {g_cfg['name']} -> {gen_cfg['name']}, defence={args.defence}")
    clf = load_guardrail(g_cfg["model_name"], device=0)
    llm = load_generator(gen_cfg["model_name"])

    writer = JsonlWriter(Path(cfg["project"]["results_dir"]) / "end_to_end.jsonl", cfg)
    conditions = list(cfg["attack"]["conditions"])
    if hybrid_mech:
        conditions.append("C6_hybrid")

    # Cache each query's real top-100 clean BM25 results once, reused across
    # every (condition, rho) below -- this is the same validated
    # "outside-the-index" scoring shortcut used throughout the sweep
    # (src.retrievers.bm25_rank_external; validated against a real
    # re-index in tests/test_retrievers.py), not a re-ranking hack.
    print("Caching clean top-100 BM25 results per query...")
    clean_cache = {qid: bm25_top_k(bm25_index, qtext, k=100) for qid, qtext in eval_queries.items()}

    for condition in conditions:
        for rho in cfg["attack"]["rhos"]:
            successes = {}
            prompts, qids, canaries = [], [], []

            for qid, qtext in eval_queries.items():
                adv = build_adversarial(qtext, qid, condition, rho, seed=seed, hybrid_mechanism=hybrid_mech)

                # 1. RETRIEVE FOR REAL: the adversarial passage's real rank
                # among the full corpus, via the validated external scorer.
                # If it ranks inside the top k, the resulting context is
                # (the top k-1 clean passages) + (the adversarial passage) --
                # inserting one new item into a sorted top-k list and keeping
                # only the top k always displaces exactly the k-th item,
                # regardless of the new item's exact rank, so this is exact,
                # not an approximation.
                clean_scores = [s for _, s in clean_cache[qid]]
                rank, _ = bm25_rank_external(qtext, adv.text, clean_scores, bm25_index.corpus_stats)
                if rank > k:
                    successes[qid] = False  # not retrieved into context -> cannot succeed
                    continue
                clean_top_k_minus_1 = [text_by_id[d] for d, _ in clean_cache[qid][: k - 1]]
                context_texts = clean_top_k_minus_1 + [adv.text]

                # 2. DEFENCE + GUARDRAIL FILTER: sanitise every passage, then
                # drop any the guardrail flags as malicious.
                sanitised = [sanitise(t, args.defence).text for t in context_texts]
                g_results = score_passages(clf, sanitised, g_cfg["malicious_label"])
                kept = [t for t, r in zip(sanitised, g_results) if r.predicted_label != g_cfg["malicious_label"]]
                if not kept:
                    successes[qid] = False
                    continue

                prompts.append(build_prompt(qtext, kept))
                qids.append(qid)
                canaries.append(adv.canary)

            if prompts:
                outputs = generate_batch(llm, prompts, max_tokens=cfg["generation"]["max_tokens"],
                                          temperature=cfg["generation"]["temperature"],
                                          supports_system_role=gen_cfg.get("supports_system_role", True))
                gen_results = judge_batch(qids, outputs, canaries)
                for r in gen_results:
                    successes[r.query_id] = r.success

            asr = asr_measured(list(successes.values()))
            print(f"{condition} rho={rho} defence={args.defence}: measured ASR_e2e = {asr:.3f}")
            writer.write({
                "retriever": "bm25", "guardrail": g_cfg["name"], "generator": gen_cfg["name"],
                "condition": condition, "rho": rho, "defence": args.defence,
                "asr_measured": asr, "success_by_query": successes,
            })

    print(f"Wrote {writer.path}")


if __name__ == "__main__":
    main()
