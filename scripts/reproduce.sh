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
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-}"
JUDGE_MODEL_PIN_FILE="${JUDGE_MODEL_PIN_FILE:-configs/judge_model.txt}"
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

FIXED_JUDGE_MODEL=""
if [[ -f "$JUDGE_MODEL_PIN_FILE" ]]; then
  FIXED_JUDGE_MODEL="$(grep -Ev '^\s*($|#)' "$JUDGE_MODEL_PIN_FILE" | head -n 1 | tr -d '[:space:]')"
fi

SAFE_MODEL_TAG="${MODEL//\//_}"
SAFETY_RESPONSES="$OUT_DIR/responses_${SAFE_MODEL_TAG}_${DATE_TAG}.jsonl"
SAFETY_JUDGMENTS="$OUT_DIR/judgments_${SAFE_MODEL_TAG}_${DATE_TAG}.jsonl"
VALUES_RESPONSES="$OUT_DIR/values_responses_${SAFE_MODEL_TAG}_${DATE_TAG}.jsonl"
SAFETY_REPORT_JSON="$OUT_DIR/safety_report_${SAFE_MODEL_TAG}_${DATE_TAG}.json"
VALUES_REPORT_JSON="$OUT_DIR/values_report_${SAFE_MODEL_TAG}_${DATE_TAG}.json"

RUN_ARGS=(--model "$MODEL" --temperature "$TEMPERATURE")
if [[ -n "$SEED" ]]; then
  RUN_ARGS+=(--seed "$SEED")
fi

export OPENAI_API_KEY

if [[ "$RUN_JUDGE" == "1" ]]; then
  if [[ -n "$FIXED_JUDGE_MODEL" ]]; then
    if [[ -z "$JUDGE_MODEL" ]]; then
      JUDGE_MODEL="$FIXED_JUDGE_MODEL"
    elif [[ "$JUDGE_MODEL" != "$FIXED_JUDGE_MODEL" ]]; then
      echo "Error: JUDGE_MODEL ($JUDGE_MODEL) differs from fixed judge model ($FIXED_JUDGE_MODEL)." >&2
      echo "To keep evaluations comparable, use the fixed judge or update $JUDGE_MODEL_PIN_FILE intentionally." >&2
      exit 1
    fi
  fi

  if [[ -z "$JUDGE_MODEL" ]]; then
    echo "Error: JUDGE_MODEL is required when RUN_JUDGE=1." >&2
    exit 1
  fi
fi

echo "Config: model=$MODEL judge=${JUDGE_MODEL:-none} fixed_judge=${FIXED_JUDGE_MODEL:-none} run_judge=$RUN_JUDGE temp=$TEMPERATURE seed=${SEED:-none}"

echo "[1/5] Running safety generation..."
"$PYTHON_BIN" src/run_eval.py \
  "${RUN_ARGS[@]}" \
  --input data/prompts.jsonl \
  --output "$SAFETY_RESPONSES"

if [[ "$RUN_JUDGE" == "1" ]]; then
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
  "${RUN_ARGS[@]}" \
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
