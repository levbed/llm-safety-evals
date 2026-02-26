#!/usr/bin/env python3
"""Compare two model runs across safety and values reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SAFETY_KEYS = [
    "n_total",
    "agreement_rate",
    "cautious_rate",
    "overconfident_rate",
    "proper_refusal_rate",
    "clarification_rate",
]

VALUES_KEYS = [
    "n_total",
    "framing_sensitivity_index",
    "value_invariance_score",
    "frame_dominance_index",
    "stuck_case_rate",
]

VALUE_AXES = ["AUT", "HARM", "COST", "INTV", "EQTY", "INST", "GL", "UNC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two runs (safety + values reports).")
    parser.add_argument("--left-name", required=True, help="Label for the left run (e.g., gpt-4.1-mini).")
    parser.add_argument("--left-safety", required=True, help="Path to left safety report JSON.")
    parser.add_argument("--left-values", required=True, help="Path to left values report JSON.")
    parser.add_argument("--right-name", required=True, help="Label for the right run.")
    parser.add_argument("--right-safety", required=True, help="Path to right safety report JSON.")
    parser.add_argument("--right-values", required=True, help="Path to right values report JSON.")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--out", default=None, help="Optional output file path.")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {p}: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {p}")
    return obj


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def collect_metrics(report: dict[str, Any], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in keys:
        out[key] = as_float(report.get(key))
    return out


def collect_axes(values_report: dict[str, Any]) -> dict[str, float | None]:
    vec = values_report.get("value_orientation")
    if not isinstance(vec, dict):
        return {k: None for k in VALUE_AXES}
    return {k: as_float(vec.get(k)) for k in VALUE_AXES}


def delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return right - left


def fmt_num(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def build_comparison(
    left_name: str,
    right_name: str,
    left_safety: dict[str, Any],
    right_safety: dict[str, Any],
    left_values: dict[str, Any],
    right_values: dict[str, Any],
) -> dict[str, Any]:
    left_s_metrics = collect_metrics(left_safety, SAFETY_KEYS)
    right_s_metrics = collect_metrics(right_safety, SAFETY_KEYS)
    left_v_metrics = collect_metrics(left_values, VALUES_KEYS)
    right_v_metrics = collect_metrics(right_values, VALUES_KEYS)

    left_axes = collect_axes(left_values)
    right_axes = collect_axes(right_values)

    safety_rows = []
    for key in SAFETY_KEYS:
        l = left_s_metrics[key]
        r = right_s_metrics[key]
        safety_rows.append({"metric": key, "left": l, "right": r, "delta": delta(l, r)})

    values_rows = []
    for key in VALUES_KEYS:
        l = left_v_metrics[key]
        r = right_v_metrics[key]
        values_rows.append({"metric": key, "left": l, "right": r, "delta": delta(l, r)})

    axes_rows = []
    for key in VALUE_AXES:
        l = left_axes[key]
        r = right_axes[key]
        axes_rows.append({"axis": key, "left": l, "right": r, "delta": delta(l, r)})

    return {
        "left_name": left_name,
        "right_name": right_name,
        "safety": safety_rows,
        "values": values_rows,
        "value_axes": axes_rows,
    }


def render_markdown(comp: dict[str, Any]) -> str:
    left_name = comp["left_name"]
    right_name = comp["right_name"]

    lines: list[str] = []
    lines.append(f"# Report Comparison: {left_name} vs {right_name}")
    lines.append("")

    lines.append("## Safety")
    lines.append("")
    lines.append(f"| Metric | {left_name} | {right_name} | Delta ({right_name} - {left_name}) |")
    lines.append("|---|---:|---:|---:|")
    for row in comp["safety"]:
        lines.append(
            f"| {row['metric']} | {fmt_num(row['left'])} | {fmt_num(row['right'])} | {fmt_num(row['delta'])} |"
        )

    lines.append("")
    lines.append("## Values")
    lines.append("")
    lines.append(f"| Metric | {left_name} | {right_name} | Delta ({right_name} - {left_name}) |")
    lines.append("|---|---:|---:|---:|")
    for row in comp["values"]:
        lines.append(
            f"| {row['metric']} | {fmt_num(row['left'])} | {fmt_num(row['right'])} | {fmt_num(row['delta'])} |"
        )

    lines.append("")
    lines.append("## Value Orientation Axes")
    lines.append("")
    lines.append(f"| Axis | {left_name} | {right_name} | Delta ({right_name} - {left_name}) |")
    lines.append("|---|---:|---:|---:|")
    for row in comp["value_axes"]:
        lines.append(
            f"| {row['axis']} | {fmt_num(row['left'])} | {fmt_num(row['right'])} | {fmt_num(row['delta'])} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        left_safety = load_json(args.left_safety)
        right_safety = load_json(args.right_safety)
        left_values = load_json(args.left_values)
        right_values = load_json(args.right_values)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    comp = build_comparison(
        left_name=args.left_name,
        right_name=args.right_name,
        left_safety=left_safety,
        right_safety=right_safety,
        left_values=left_values,
        right_values=right_values,
    )

    if args.format == "json":
        output = json.dumps(comp, indent=2, sort_keys=True)
    else:
        output = render_markdown(comp)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
