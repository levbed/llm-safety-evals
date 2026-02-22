# llm-safety-evals

An MVP safety eval harness for LLM behavior on risky/ambiguous prompts. It runs a target model on `data/prompts.jsonl`, labels each response with an LLM-as-judge, and computes aggregate rates for cautiousness, overconfidence, refusal quality, and clarification behavior.

It now includes a VIM-style values extension for measuring framing-sensitive value orientation drift. Here, VIM refers to the value-orientation measurement framework for surfacing hidden value trade-offs in clinical AI decisions as defined in the NEJM AI publication: https://ai.nejm.org/doi/full/10.1056/AIp2501266.

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
MODEL=<MODEL> JUDGE_MODEL=<JUDGE_MODEL> ./scripts/reproduce.sh
```

Optional smoke run (skip judge stage):
```bash
MODEL=<MODEL> RUN_JUDGE=0 ./scripts/reproduce.sh
```

Artifacts are saved under `outputs/` with UTC timestamp suffixes.

## Manual Run

### Safety track

1. Generate responses:
```bash
python src/run_eval.py --model <MODEL>
```

2. Judge responses:
```bash
python src/judge.py --judge-model <JUDGE_MODEL>
```

3. Compute safety metrics:
```bash
python src/metrics.py
```

### Values track (VIM-style extension)

1. Generate values responses:
```bash
python src/run_eval.py \
  --model <MODEL> \
  --input data/values_prompts.jsonl \
  --output outputs/values_responses.jsonl
```

2. Compute values metrics:
```bash
python src/value_metrics.py --responses outputs/values_responses.jsonl
```

`value_metrics.py` reports:
- Value Orientation Vector (AUT/HARM/COST/INTV/EQTY/INST/GL/UNC)
- Framing Sensitivity Index (mean pairwise L1 distance across NEUTRAL/PROVIDER/PATIENT frames)
- Per-case frame-distance summaries

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
