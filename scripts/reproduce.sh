#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
RUN_JUDGE="${RUN_JUDGE:-1}"
OUT_DIR="${OUT_DIR:-outputs}"
PYTHON_BIN="${PYTHON_BIN:-}"
DATE_TAG="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: python interpreter not found (tried python3, python)." >&2
    exit 1
  fi
fi

if [[ -z "$MODEL" ]]; then
  echo "Error: MODEL is required. Example: MODEL=gpt-4.1-mini" >&2
  exit 1
fi

if [[ -z "$OPENAI_API_KEY" ]]; then
  echo "Error: OPENAI_API_KEY is required." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

SAFE_MODEL_TAG="${MODEL//\//_}"
SAFETY_RESPONSES="$OUT_DIR/responses_${SAFE_MODEL_TAG}_${DATE_TAG}.jsonl"
SAFETY_JUDGMENTS="$OUT_DIR/judgments_${SAFE_MODEL_TAG}_${DATE_TAG}.jsonl"
VALUES_RESPONSES="$OUT_DIR/values_responses_${SAFE_MODEL_TAG}_${DATE_TAG}.jsonl"
SAFETY_REPORT_JSON="$OUT_DIR/safety_report_${SAFE_MODEL_TAG}_${DATE_TAG}.json"
VALUES_REPORT_JSON="$OUT_DIR/values_report_${SAFE_MODEL_TAG}_${DATE_TAG}.json"

export OPENAI_API_KEY

echo "[1/5] Running safety generation..."
"$PYTHON_BIN" src/run_eval.py \
  --model "$MODEL" \
  --input data/prompts.jsonl \
  --output "$SAFETY_RESPONSES"

if [[ "$RUN_JUDGE" == "1" ]]; then
  if [[ -z "$JUDGE_MODEL" ]]; then
    echo "Error: JUDGE_MODEL is required when RUN_JUDGE=1." >&2
    exit 1
  fi
  echo "[2/5] Running safety judge..."
  "$PYTHON_BIN" src/judge.py \
    --judge-model "$JUDGE_MODEL" \
    --responses "$SAFETY_RESPONSES" \
    --output "$SAFETY_JUDGMENTS"

  echo "[3/5] Computing safety metrics..."
  "$PYTHON_BIN" src/metrics.py \
    --responses "$SAFETY_RESPONSES" \
    --judgments "$SAFETY_JUDGMENTS" \
    --format json > "$SAFETY_REPORT_JSON"
else
  echo "[2/5] Skipping safety judge/metrics (RUN_JUDGE=$RUN_JUDGE)."
fi

echo "[4/5] Running values generation..."
"$PYTHON_BIN" src/run_eval.py \
  --model "$MODEL" \
  --input data/values_prompts.jsonl \
  --output "$VALUES_RESPONSES"

echo "[5/5] Computing values metrics..."
"$PYTHON_BIN" src/value_metrics.py \
  --responses "$VALUES_RESPONSES" \
  --format json > "$VALUES_REPORT_JSON"

echo "Done. Artifacts:"
echo "- safety responses: $SAFETY_RESPONSES"
if [[ "$RUN_JUDGE" == "1" ]]; then
  echo "- safety judgments: $SAFETY_JUDGMENTS"
  echo "- safety report: $SAFETY_REPORT_JSON"
fi
echo "- values responses: $VALUES_RESPONSES"
echo "- values report: $VALUES_REPORT_JSON"
