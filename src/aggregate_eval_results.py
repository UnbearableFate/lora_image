#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AggregateSpec:
    key_columns: tuple[str, ...]
    metric_column: str


def _expand_inputs(inputs: list[str]) -> list[str]:
    paths: list[str] = []
    for item in inputs:
        matches = glob.glob(item)
        if matches:
            paths.extend(matches)
        else:
            paths.append(item)
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            deduped.append(norm)
    return deduped


def _read_rows(paths: Iterable[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV (no header): {path}")
            for row in reader:
                row["__source_path"] = path
                rows.append(row)
    return rows


def _validate_columns(rows: list[dict[str, str]], spec: AggregateSpec) -> None:
    if not rows:
        raise ValueError("No rows found in input CSV(s).")
    sample = rows[0]
    missing = [c for c in (*spec.key_columns, spec.metric_column) if c not in sample]
    if missing:
        available = ", ".join(sorted(sample.keys()))
        raise ValueError(f"Missing columns: {missing}. Available columns: {available}")


def _to_float(value: str, *, context: str) -> float:
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"Failed to parse float for {context}: {value!r}") from e


def aggregate(rows: list[dict[str, str]], spec: AggregateSpec) -> list[dict[str, str]]:
    _validate_columns(rows, spec)

    grouped: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        key = tuple((row.get(col) or "").strip() for col in spec.key_columns)
        metric = _to_float((row.get(spec.metric_column) or "").strip(), context=spec.metric_column)
        grouped.setdefault(key, []).append(metric)

    computed: list[tuple[float, dict[str, str]]] = []
    for key, values in grouped.items():
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) >= 2 else 0.0
        out: dict[str, str] = {col: val for col, val in zip(spec.key_columns, key, strict=True)}
        out[f"{spec.metric_column}_mean"] = f"{mean:.10g}"
        out[f"{spec.metric_column}_std"] = f"{std:.10g}"
        out[f"{spec.metric_column}_n"] = str(len(values))
        computed.append((mean, out))

    computed.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in computed]


def _write_csv(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No output rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate eval_results.csv across different seeds by grouping on config columns, "
            "then computing mean/std for metric_accuracy."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more CSV paths or glob patterns (e.g. experiments/*/*/eval_results.csv).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output CSV path (default: same directory as the first expanded input, "
            "with suffix '_avg.csv')."
        ),
    )
    parser.add_argument(
        "--key-cols",
        default="base_model,dataset_name,init_lora_weights,extra,r,lora_alpha",
        help="Comma-separated columns used to group rows (default: %(default)s).",
    )
    parser.add_argument(
        "--metric-col",
        default="metric_accuracy",
        help="Metric column to aggregate (default: %(default)s).",
    )
    args = parser.parse_args()

    spec = AggregateSpec(
        key_columns=tuple(c.strip() for c in args.key_cols.split(",") if c.strip()),
        metric_column=args.metric_col.strip(),
    )

    paths = _expand_inputs(args.inputs)
    if not paths:
        raise ValueError("No input files found after glob expansion.")
    if args.out:
        out_path = args.out
    else:
        first = paths[0]
        first_dir = os.path.dirname(os.path.abspath(first))
        first_stem = os.path.splitext(os.path.basename(first))[0]
        out_path = os.path.join(first_dir, f"{first_stem}_avg.csv")
    rows = _read_rows(paths)
    out_rows = aggregate(rows, spec)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    _write_csv(out_path, out_rows)


if __name__ == "__main__":
    main()
