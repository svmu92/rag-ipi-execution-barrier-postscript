# rag-ipi-execution-barrier-postscript

Code for the postscript measurement in "Benchmarking the Retrieval-Evasion
Trade-off of Obfuscated IPI in RAG" (COMPSCI 764, Group 28), Section VI-G,
"Measuring the execution barrier P_gen (postscript)".

## What this measures

The report's headline risk numbers are all `ASR* = P_ret * epsilon`, an
explicit upper bound, because the execution barrier `P_gen` needs a GPU the
main sweep did not have (it ran on a CPU-only laptop). This repo fills that
one gap for the hybrid trigger/payload condition -- the only condition with
non-trivial `ASR*` in the report (up to 0.17 at rho=1; every single-family
carrier stayed near zero) -- using a single consumer GPU (one 16GB T4).

Two scripts, both against the same obfuscation/retrieval/guardrail/canary
machinery:

- `scripts/06_end_to_end.py` -- a real, joint measurement of
  `ASR_e2e(rho)`: real BM25 retrieval (the adversarial passage only reaches
  the generator if it is actually retrieved into the top-k context), real
  guardrail filtering, real generation, judged by exact canary match. This
  found `ASR_e2e = 0.000` at every rho, despite the driver log showing the
  adversarial passage reaching the generation step 27-29/30 times at every
  rho -- ruling out a retrieval or guardrail bug as the explanation.
- `scripts/08_pgen_direct_feed_probe.py` -- the follow-up that explains the
  flat zero above: it bypasses retrieval and the guardrail entirely and
  hands the adversarial passage to the generator as its sole context, for 3
  queries at rho in {0, 0.25, 0.5, 1}. Result: compliance fell from 3/3 at
  rho=0 (a positive control -- not blanket refusal) to 0/3 at rho=1, with
  the breakdown around rho=0.5. Read together, the two scripts show that
  evasion and generator legibility move against the attacker from opposite
  ends of the same rho range -- low rho is legible enough for the generator
  but also legible enough for the guardrail; high rho evades the guardrail
  but is no longer legible to the generator -- so `ASR_e2e ~ 0` throughout
  despite a non-trivial `ASR*`.

See the report's Table XI and Section VI-G for the exact numbers this is
meant to reproduce.

## Relationship to the rest of the project

This is a standalone, self-contained extract -- not the full project
codebase, and not the same codebase as the `rag-obfuscation-benchmark`
GitHub repo the report's footnote links as "the full implementation". It
exists specifically so the GPU-dependent postscript measurement has its own
clean, runnable, minimal record, independent of either of those.

`src/generate.py`'s `load_generator` is the one piece of code here that
differs from how generation would run on larger hardware: it defaults to a
public 4-bit AWQ checkpoint (`solidrust/Meta-Llama-3-8B-Instruct-hf-AWQ`) at
`dtype="half"`, pinned to a second GPU when one is visible, because a single
16GB T4 cannot fit the full-precision model and does not support bfloat16.
See that function's docstring for the full reasoning, and pass
`quantization=None` with the original `meta-llama/Meta-Llama-3-8B-Instruct`
on a larger/newer GPU to skip all of it.

Everything else -- the six obfuscation operators, the normalisation
defence, BM25 retrieval, the guardrail wrapper, the canary protocol -- is
unmodified.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install vllm                   # first: it pins its own compatible torch/CUDA build
pip install -r requirements.txt
huggingface-cli login              # still needed for HF downloads generally
```

Needs a GPU with >= 16GB memory. On a Turing-class GPU (compute capability
7.5, e.g. a T4), bitsandbytes on-the-fly quantisation is not a registered
vLLM quantisation method on some vLLM builds (0.31.0 here) -- if
`quantization="awq"` also fails to load on your vLLM version, check for a
`vllm-bnb-plugin`-style fix for your version, or fall back to CPU-only
`--skip-gpu` checks while you sort it out.

## Run

```bash
python scripts/00_check_environment.py              # confirm the GPU/model/guardrail load
python scripts/01_prepare_data.py --config configs/postscript.yaml   # downloads BEIR NQ, samples 30 queries
python scripts/06_end_to_end.py --config configs/postscript.yaml     # real joint ASR_e2e(rho), hybrid only
python scripts/08_pgen_direct_feed_probe.py --config configs/postscript.yaml  # the direct-feed probe
```

Results land under `results/nq/`: `end_to_end.jsonl` (one line per
(condition, rho) pair, each with a per-query `success_by_query` breakdown
and the config + git-commit provenance that produced it) and
`execution_postscript_probe.jsonl` (one line per probed query).

`configs/postscript.yaml` is the scoped config this actually ran with: 30
queries (not the main sweep's 200), one seed (not three), BM25 only, and
only the hybrid condition (`attack.conditions: []`, `attack.hybrid_mechanism:
homoglyph`) -- the single-family conditions are not covered here since the
report's main sweep already shows they stay near zero risk.
`configs/default.yaml` is kept alongside it as the un-scoped shape these
overrides were made from.

## Caveats

These numbers are indicative, not a statistically matched replacement for
the main sweep: n=30 vs. the main sweep's n=200, one seed vs. three, one
retriever (BM25) only, and (in the direct-feed probe) no bootstrap CI given
the sample size. The generator is a third-party AWQ repack of
Llama-3-8B-Instruct, not the full-precision weights used nominally
elsewhere in the project, so results are scoped to that checkpoint. The
rho~0.5 breakdown point in the probe rests on a single query flipping
outcome in a three-query sample and should be read as approximate, not a
precise threshold.
