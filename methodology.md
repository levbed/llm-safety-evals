# Methodology

## Goal

This repository measures two distinct properties of model behavior:
- Safety behavior on risky or ambiguous prompts.
- Value orientation drift under framing changes (VIM-style extension).

The objective is not to determine a single "morally correct" answer. The objective is to make behavioral differences measurable, reproducible, and comparable across models and runs.

## Evaluation Tracks

### 1) Safety Track

Files:
- `data/prompts.jsonl`
- `src/run_eval.py`
- `src/judge.py`
- `src/metrics.py`

Process:
1. Generate model outputs for each prompt.
2. Label outputs with an LLM-as-judge on four binary labels:
   - `cautious`
   - `overconfident`
   - `proper_refusal`
   - `asked_clarification`
3. Aggregate rates and behavior-agreement metrics.

Key assumptions:
- LLM judge labels are a proxy for rubric compliance, not ground truth.
- Prompt set is intentionally small for rapid iteration.

### 2) Values Track (VIM-style)

Files:
- `data/values_prompts.jsonl`
- `src/run_eval.py`
- `src/value_metrics.py`

Design:
- Each scenario appears in three framing conditions:
  - `NEUTRAL`
  - `PROVIDER`
  - `PATIENT`
- Responses are scored across 8 value dimensions:
  - `AUT`, `HARM`, `COST`, `INTV`, `EQTY`, `INST`, `GL`, `UNC`

Metrics:
- Value Orientation Vector (mean lexical score by dimension).
- Framing Sensitivity Index (mean pairwise L1 distance across frame-specific responses).
- Per-case frame-distance summaries.

## Threat Model and Failure Modes

What this benchmark can detect:
- Relative shifts in behavior under role/framing pressure.
- High-level changes in refusal/caution patterns between model versions.
- Potentially unstable value orientation signatures.

What this benchmark cannot guarantee:
- Clinical correctness of advice.
- Normative correctness of trade-off decisions.
- Causal attribution of values to model internals.

Known failure modes:
- Lexical scoring can be gamed by keyword stuffing.
- LLM-as-judge can inherit the judge model's own biases.
- Prompt leakage and benchmark familiarity may inflate scores.
- Small sample size increases variance and overfitting risk.

## Reproducibility Protocol

Recommended controls for published results:
- Fix model identifiers explicitly.
- Fix temperature and seed when supported.
- Run at least 3 repeated seeds for stability checks.
- Store raw JSONL artifacts for generation and judgments.
- Report date, model version, and command lines used.

Suggested minimum release artifacts:
- `outputs/responses_*.jsonl`
- `outputs/judgments_*.jsonl`
- `reports/baseline_results.md`
- `reports/summary_*.json`

## Validation Roadmap (to move from MVP to research-grade)

1. Expand both datasets (100+ prompts per track).
2. Add train/dev/test split and freeze test set.
3. Add second scorer for values (LLM judge rubric) and measure scorer agreement.
4. Add confidence intervals via bootstrap.
5. Add blinded human review subset for calibration.
6. Add regression thresholds for CI gating.

## Reporting Standards for Public Claims

When publishing results from this repo:
- Clearly label this as a lightweight benchmark harness.
- Include limitations and failure modes.
- Avoid claiming normative alignment guarantees.
- Report uncertainty and run-to-run variance where possible.
