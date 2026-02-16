# llm-safety-evals

## How To Run The Runner

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Set your API key:
```bash
export OPENAI_API_KEY="your_api_key_here"
```
3. Run eval:
```bash
python src/run_eval.py --model <MODEL_NAME>
```

### Useful Options

- `--input data/prompts.jsonl` (default)
- `--output outputs/responses.jsonl` (default)
- `--max-prompts 3` to process only 3 uncached prompts
- `--force` to re-run prompts regardless of cache
- `--sleep 0.2` seconds between calls (default)
- `--temperature 0`
- `--seed <int>` (optional)

### Behavior Notes

- Input JSONL is validated line-by-line; malformed JSON or duplicate prompt ids in input cause non-zero exit.
- Without `--force`, cached `prompt_id`s found in the output file are skipped and no model call is made.
- With `--force`, the output file is atomically deduplicated by `prompt_id` on each successful write.
- The script exits `0` for normal runs (even with per-prompt failures) and prints a summary with completed/skipped/failed counts.
