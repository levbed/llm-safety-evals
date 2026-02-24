# llm-safety-evals

An MVP safety eval harness for LLM behavior on risky/ambiguous prompts. It runs a target model on `data/prompts.jsonl`, labels each response with an LLM-as-judge, and computes aggregate rates for cautiousness, overconfidence, refusal quality, and clarification behavior.

It includes a VIM-style values extension for measuring framing-sensitive value orientation drift. Here, VIM refers to the value-orientation measurement framework for surfacing hidden value trade-offs in clinical AI decisions as defined in the NEJM AI publication: https://ai.nejm.org/doi/full/10.1056/AIp2501266.

Important: the axis-level lexical scoring in this repository is a VIM-inspired, repository-specific heuristic proxy (not a verbatim scoring standard from the publication).

Note: this repository is an MVP and an evolving benchmark; prompts, scoring, and reporting will be expanded and refined over time.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set credentials:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

3. Run the reproducible pipeline:
```bash
MODEL=<MODEL> JUDGE_MODEL=<JUDGE_MODEL> TEMPERATURE=0 SEED=42 ./scripts/reproduce.sh
```

Optional smoke run (skip judge stage):
```bash
MODEL=<MODEL> TEMPERATURE=0 SEED=42 RUN_JUDGE=0 ./scripts/reproduce.sh
```

Artifacts are saved under `outputs/` with UTC timestamp suffixes.

Reproducibility knobs:
- `TEMPERATURE` (default `0.0`)
- `SEED` (optional, forwarded to `run_eval.py`)

## Manual Run

### Safety track

1. Generate responses:
```bash
python3 src/run_eval.py --model <MODEL>
```

2. Judge responses:
```bash
python3 src/judge.py --judge-model <JUDGE_MODEL>
```

3. Compute safety metrics:
```bash
python3 src/metrics.py
```

### Values track (VIM-style extension)

1. Generate values responses:
```bash
python3 src/run_eval.py \
  --model <MODEL> \
  --input data/values_prompts.jsonl \
  --output outputs/values_responses.jsonl
```

2. Compute values metrics:
```bash
python3 src/value_metrics.py --responses outputs/values_responses.jsonl
```

`value_metrics.py` reports:
- Value Orientation Vector (AUT/HARM/COST/INTV/EQTY/INST/GL/UNC)
- Framing Sensitivity Index (mean pairwise L1 distance across NEUTRAL/PROVIDER/PATIENT frames)
- Value Invariance Score (1 - normalized FSI; higher means less change across frames)
- Frame Dominance Index and Stuck Case Rate (signals value-axis lock-in across frames)
- Per-case frame-distance and dominant-axis summaries

## Public Reporting Pack

- Methodology and limitations: `methodology.md`
- Reproducible runner: `scripts/reproduce.sh`
- Baseline report template: `reports/baseline_template.md`

## Notes

- `src/run_eval.py` validates input JSONL and stops on malformed JSON or duplicate prompt IDs.
- Runner caching: without `--force`, existing `prompt_id` entries in output are skipped.
- Judge caching: without `--force`, existing `prompt_id` judgments are skipped.
- `src/judge.py` writes judgments via atomic upsert, guaranteeing one line per `prompt_id`.
- `outputs/` is gitignored by default, so publish selected reports under `reports/` when needed.
