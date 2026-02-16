# llm-safety-evals

Micro-MVP safety eval harness for LLM behavior on risky/ambiguous prompts. It runs a target model on `data/prompts.jsonl`, labels each response with an LLM-as-judge, and computes aggregate rates for cautiousness, overconfidence, refusal quality, and clarification behavior. All pipeline artifacts are JSONL and cache-aware so interrupted runs can resume safely.

## How to run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set credentials:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

3. Run generation:
```bash
python src/run_eval.py --model <MODEL>
```

4. Run judging:
```bash
python src/judge.py --judge-model <JUDGE_MODEL>
```

5. Compute metrics:
```bash
python src/metrics.py
```

## Notes

- `src/run_eval.py` validates input JSONL and **stops on malformed JSON** or **duplicate prompt ids** in input.
- Runner caching: without `--force`, existing `prompt_id` entries in output are skipped (no extra model calls).
- Judge caching: without `--force`, existing `prompt_id` judgments are skipped.
- `src/judge.py` writes judgments via atomic upsert, guaranteeing one line per `prompt_id` in `outputs/judgments.jsonl`.
