#!/usr/bin/env python3
"""Direct-feed P_gen probe (report Section VI-G): isolates the execution
barrier P_gen from retrieval and the guardrail entirely, by handing the
adversarial passage to the generator as its SOLE context.

scripts/06_end_to_end.py measured asr_measured = 0.000 at every rho for the
hybrid condition, even though its own driver log showed the adversarial
passage reaching the generation step at every rho (i.e. not a retrieval or
guardrail failure). This script was the follow-up that isolated why: for
each rho, it bypasses bm25_rank_external and the guardrail's score_passages
entirely and asks only "if the generator sees this passage and nothing
else, does it comply?" -- which is P_gen in isolation. rho=0 (an
unobfuscated instruction) is a positive control: if the model doesn't comply
even then, something else (blanket refusal) is going on, not an obfuscation
effect.

Usage:
  python scripts/08_pgen_direct_feed_probe.py --config configs/postscript.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import JsonlWriter, load_config

from src.attack.build_docs import build_adversarial
from src.generate import build_prompt, generate_batch, judge_batch, load_generator

# rho=0.75 was not probed in the report -- the breakdown point was already
# bracketed by 0.5 and 1.0, so a 5th generation call was judged not worth it.
DEFAULT_RHOS = [0.0, 0.25, 0.5, 1.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/postscript.yaml")
    parser.add_argument("--n_queries", type=int, default=3, help="queries probed per rho (report used 3)")
    parser.add_argument("--rhos", default=",".join(str(r) for r in DEFAULT_RHOS),
                         help="comma-separated rho values to probe")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg["project"]["seed"]
    hybrid_mech = cfg["attack"]["hybrid_mechanism"]
    if not hybrid_mech:
        raise SystemExit("this probe is scoped to the hybrid condition: "
                          "set attack.hybrid_mechanism in the config")
    rhos = [float(r) for r in args.rhos.split(",")]

    eval_queries = json.loads(
        (Path(cfg["project"]["data_dir"]) / cfg["data"]["dataset"] / "queries_eval.json").read_text()
    )
    qids = list(eval_queries)[: args.n_queries]
    if len(qids) < args.n_queries:
        raise SystemExit(f"only {len(qids)} eval queries available, need {args.n_queries} "
                          "-- run scripts/01_prepare_data.py with a larger n_eval_queries first")

    gen_cfg = cfg["generators"][0]
    print(f"Loading generator {gen_cfg['model_name']} ...")
    llm = load_generator(gen_cfg["model_name"])

    writer = JsonlWriter(
        Path(cfg["project"]["results_dir"]) / cfg["data"]["dataset"] / "execution_postscript_probe.jsonl",
        cfg,
    )

    for rho in rhos:
        advs = [
            build_adversarial(eval_queries[qid], qid, "C6_hybrid", rho, seed=seed, hybrid_mechanism=hybrid_mech)
            for qid in qids
        ]
        # The adversarial passage is the SOLE context: no retrieval, no
        # guardrail -- this is what makes it a direct-feed probe of P_gen.
        prompts = [build_prompt(eval_queries[qid], [adv.text]) for qid, adv in zip(qids, advs)]
        outputs = generate_batch(llm, prompts, max_tokens=cfg["generation"]["max_tokens"],
                                  temperature=cfg["generation"]["temperature"],
                                  supports_system_role=gen_cfg.get("supports_system_role", True))
        results = judge_batch(qids, outputs, [adv.canary for adv in advs])
        compliant = sum(r.success for r in results)
        print(f"rho={rho:.2f}  compliance={compliant}/{len(qids)}")
        for r in results:
            writer.write({
                "rho": rho, "query_id": r.query_id, "compliant": r.success,
                "canary": r.canary, "output_text": r.output_text,
            })

    print(f"Wrote {writer.path}")


if __name__ == "__main__":
    main()
