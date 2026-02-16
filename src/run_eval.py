#!/usr/bin/env python3
"""Run a micro safety eval by calling a model for each prompt in a JSONL dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        InternalServerError,
        OpenAI,
        RateLimitError,
    )
except ImportError:  # pragma: no cover - exercised in runtime, not tests
    OpenAI = None  # type: ignore[assignment]
    APIConnectionError = APITimeoutError = RateLimitError = InternalServerError = APIStatusError = Exception  # type: ignore[assignment]


ALLOWED_CATEGORIES = {"biomedical_risk", "uncertainty", "boundary"}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}
ALLOWED_AMBIGUITY_LEVELS = {"low", "medium", "high"}
ALLOWED_EXPECTED_BEHAVIORS = {
    "cautious_answer",
    "refuse",
    "express_uncertainty",
    "ask_clarifying_question",
}
RETRY_DELAYS_SECONDS = [1, 2, 4, 8, 16]


@dataclass(frozen=True)
class PromptRecord:
    id: str
    category: str
    risk_level: str
    ambiguity_level: str
    expected_behavior: str
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model eval on prompt dataset.")
    parser.add_argument("--model", required=True, help="Model name to call.")
    parser.add_argument("--input", default="data/prompts.jsonl", help="Input prompts JSONL path.")
    parser.add_argument(
        "--output",
        default="outputs/responses.jsonl",
        help="Output responses JSONL path.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Maximum number of uncached prompts to run this invocation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-run prompts. Output file is deduplicated by prompt_id.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between successful model calls.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to the model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed parameter for compatible models.",
    )
    return parser.parse_args()


def _expect_string(obj: dict[str, Any], field: str, line_no: int) -> str:
    value = obj.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Invalid {field!r} at line {line_no}: expected string.")
    return value


def load_prompts(path: Path) -> list[PromptRecord]:
    if not path.exists():
        raise ValueError(f"Input file not found: {path}")

    prompts: list[PromptRecord] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at line {line_no}: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid JSON object at line {line_no}: expected object.")

            prompt_id = _expect_string(obj, "id", line_no)
            if prompt_id in seen_ids:
                raise ValueError(f"Duplicate prompt id at line {line_no}: {prompt_id!r}")
            seen_ids.add(prompt_id)

            category = _expect_string(obj, "category", line_no)
            risk_level = _expect_string(obj, "risk_level", line_no)
            ambiguity_level = _expect_string(obj, "ambiguity_level", line_no)
            expected_behavior = _expect_string(obj, "expected_behavior", line_no)
            prompt_text = _expect_string(obj, "prompt", line_no)

            if category not in ALLOWED_CATEGORIES:
                raise ValueError(f"Invalid category at line {line_no}: {category!r}")
            if risk_level not in ALLOWED_RISK_LEVELS:
                raise ValueError(f"Invalid risk_level at line {line_no}: {risk_level!r}")
            if ambiguity_level not in ALLOWED_AMBIGUITY_LEVELS:
                raise ValueError(f"Invalid ambiguity_level at line {line_no}: {ambiguity_level!r}")
            if expected_behavior not in ALLOWED_EXPECTED_BEHAVIORS:
                raise ValueError(f"Invalid expected_behavior at line {line_no}: {expected_behavior!r}")

            prompts.append(
                PromptRecord(
                    id=prompt_id,
                    category=category,
                    risk_level=risk_level,
                    ambiguity_level=ambiguity_level,
                    expected_behavior=expected_behavior,
                    prompt=prompt_text,
                )
            )
    return prompts


def load_existing_outputs(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    if not path.exists():
        return records_by_id, order

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in output at line {line_no}: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Invalid output JSON object at line {line_no}: expected object.")

            prompt_id = obj.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"Invalid output record at line {line_no}: missing string prompt_id.")
            if prompt_id not in records_by_id:
                order.append(prompt_id)
            records_by_id[prompt_id] = obj
    return records_by_id, order


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _usage_to_dict(usage_obj: Any) -> dict[str, int] | None:
    if usage_obj is None:
        return None

    def _get(name: str) -> Any:
        if isinstance(usage_obj, dict):
            return usage_obj.get(name)
        return getattr(usage_obj, name, None)

    input_tokens = _get("input_tokens")
    output_tokens = _get("output_tokens")
    total_tokens = _get("total_tokens")
    values = (input_tokens, output_tokens, total_tokens)
    if all(isinstance(v, int) for v in values):
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }
    return None


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    parts: list[str] = []
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for piece in content:
                text = getattr(piece, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
    return "\n".join(parts)


def is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return status_code == 429 or (isinstance(status_code, int) and status_code >= 500)
    return False


def call_model_with_retries(
    client: Any,
    model: str,
    prompt_text: str,
    temperature: float,
    seed: int | None,
) -> tuple[str, dict[str, int] | None]:
    request: dict[str, Any] = {
        "model": model,
        "input": prompt_text,
        "temperature": temperature,
    }
    if seed is not None:
        request["seed"] = seed

    for attempt_idx in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            response = client.responses.create(**request)
            return extract_response_text(response), _usage_to_dict(getattr(response, "usage", None))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt_idx >= len(RETRY_DELAYS_SECONDS) or not is_transient_error(exc):
                raise
            delay = RETRY_DELAYS_SECONDS[attempt_idx]
            print(
                f"  transient error ({exc.__class__.__name__}); retrying in {delay}s...",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("Unreachable retry state.")


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def build_output_record(
    prompt: PromptRecord,
    model: str,
    response_text: str,
    usage: dict[str, int] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "prompt_id": prompt.id,
        "model": model,
        "timestamp_utc": now_iso_utc(),
        "prompt": prompt.prompt,
        "response_text": response_text,
        "meta": {
            "category": prompt.category,
            "risk_level": prompt.risk_level,
            "ambiguity_level": prompt.ambiguity_level,
            "expected_behavior": prompt.expected_behavior,
        },
    }
    if usage is not None:
        record["usage"] = usage
    return record


def main() -> int:
    args = parse_args()

    if args.max_prompts is not None and args.max_prompts < 0:
        print("Error: --max-prompts must be >= 0.", file=sys.stderr)
        return 1
    if args.sleep < 0:
        print("Error: --sleep must be >= 0.", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        prompts = load_prompts(input_path)
    except ValueError as exc:
        print(f"Input validation error: {exc}", file=sys.stderr)
        return 2

    try:
        existing_by_id, existing_order = load_existing_outputs(output_path)
    except ValueError as exc:
        print(f"Output parse error: {exc}", file=sys.stderr)
        return 1

    total_prompts = len(prompts)
    completed = 0
    failed = 0
    skipped_cached = 0

    pending: list[tuple[int, PromptRecord]] = []
    for idx, prompt in enumerate(prompts, start=1):
        if not args.force and prompt.id in existing_by_id:
            skipped_cached += 1
            print(f"[{idx}/{total_prompts}] {prompt.id} (skipped cached)", flush=True)
            continue
        pending.append((idx, prompt))

    if args.max_prompts is not None:
        pending = pending[: args.max_prompts]

    print(
        f"Running {len(pending)} prompt(s) with model={args.model!r} "
        f"(total={total_prompts}, cached_skips={skipped_cached}, force={args.force})",
        flush=True,
    )

    if pending:
        if OpenAI is None:
            print("Error: openai package is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
            return 1
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
            return 1
        client = OpenAI()
    else:
        client = None
    use_atomic_upsert = bool(args.force)

    if use_atomic_upsert:
        for idx, prompt in pending:
            try:
                response_text, usage = call_model_with_retries(
                    client=client,
                    model=args.model,
                    prompt_text=prompt.prompt,
                    temperature=args.temperature,
                    seed=args.seed,
                )
                record = build_output_record(prompt, args.model, response_text, usage)

                existing_by_id[prompt.id] = record
                if prompt.id not in existing_order:
                    existing_order.append(prompt.id)
                ordered_records = [existing_by_id[prompt_id] for prompt_id in existing_order]
                atomic_write_jsonl(output_path, ordered_records)

                completed += 1
                print(f"[{idx}/{total_prompts}] {prompt.id} (ok)", flush=True)
            except KeyboardInterrupt:
                print("\nInterrupted by user.", file=sys.stderr)
                print(
                    f"Summary: total_prompts={total_prompts} completed={completed} "
                    f"skipped_cached={skipped_cached} failed={failed}",
                    flush=True,
                )
                return 130
            except Exception as exc:
                failed += 1
                print(f"[{idx}/{total_prompts}] {prompt.id} (failed: {exc})", flush=True)

            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        with output_path.open("a", encoding="utf-8") as append_f:
            for idx, prompt in pending:
                try:
                    response_text, usage = call_model_with_retries(
                        client=client,
                        model=args.model,
                        prompt_text=prompt.prompt,
                        temperature=args.temperature,
                        seed=args.seed,
                    )
                    record = build_output_record(prompt, args.model, response_text, usage)

                    append_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    append_f.flush()
                    os.fsync(append_f.fileno())
                    existing_by_id[prompt.id] = record

                    completed += 1
                    print(f"[{idx}/{total_prompts}] {prompt.id} (ok)", flush=True)
                except KeyboardInterrupt:
                    print("\nInterrupted by user.", file=sys.stderr)
                    print(
                        f"Summary: total_prompts={total_prompts} completed={completed} "
                        f"skipped_cached={skipped_cached} failed={failed}",
                        flush=True,
                    )
                    return 130
                except Exception as exc:
                    failed += 1
                    print(f"[{idx}/{total_prompts}] {prompt.id} (failed: {exc})", flush=True)

                if args.sleep > 0:
                    time.sleep(args.sleep)

    print(
        f"Summary: total_prompts={total_prompts} completed={completed} "
        f"skipped_cached={skipped_cached} failed={failed}",
        flush=True,
    )
    if args.force:
        print(
            "Force mode note: output file is atomically deduplicated by prompt_id on each successful write.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
