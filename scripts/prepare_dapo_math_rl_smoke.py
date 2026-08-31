#!/usr/bin/env python3
"""Prepare and validate a fixed DAPO-Math smoke split for RL bring-up."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from verl.utils.reward_score import default_compute_score


def file_sha256(path: Path) -> str:
    """Return a file's SHA256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_identifier(row: Mapping[str, Any]) -> str:
    """Return a row's frozen upstream problem ID."""
    extra_info = row.get("extra_info")
    identifier = extra_info.get("index") if isinstance(extra_info, Mapping) else None
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("each row needs a nonempty extra_info.index")
    return identifier


def validate_row(row: Mapping[str, Any]) -> None:
    """Validate the fields consumed by the vLLM rollout and DAPO reward."""
    if row.get("data_source") != "math_dapo":
        raise ValueError("smoke rows must use data_source=math_dapo")
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("each row needs a nonempty chat prompt")
    for message in prompt:
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str):
            raise ValueError("each prompt message needs a string role")
        if not isinstance(message.get("content"), str) or not message["content"]:
            raise ValueError("each prompt message needs nonempty string content")
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, Mapping):
        raise ValueError("each row needs reward_model.ground_truth")
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, str) or not ground_truth:
        raise ValueError("each row needs nonempty reward_model.ground_truth")
    row_identifier(row)


def score_value(result: Any) -> float:
    """Extract the scalar score returned by VERL's reward function."""
    if isinstance(result, Mapping):
        if "score" not in result:
            raise ValueError("reward result mapping needs a score field")
        result = result["score"]
    return float(result)


def validate_chat_and_reward(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    scorer: Callable[..., Any] = default_compute_score,
) -> dict[str, Any]:
    """Exercise the exact chat template and math reward used by the smoke run."""
    rendered_digests: list[str] = []
    token_counts: list[int] = []
    for row in rows:
        validate_row(row)
        rendered = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("chat template produced no text")
        if row["prompt"][-1]["content"] not in rendered:
            raise ValueError("chat template omitted the user prompt")
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError("chat template produced no tokens")

        ground_truth = row["reward_model"]["ground_truth"]
        correct = score_value(
            scorer(row["data_source"], "Answer: " + ground_truth, ground_truth)
        )
        incorrect = score_value(
            scorer(row["data_source"], "Answer: __invalid_smoke_answer__", ground_truth)
        )
        if correct != 1.0 or incorrect != -1.0:
            raise ValueError(
                "math_dapo reward did not distinguish a correct Answer: value"
            )
        rendered_digests.append(hashlib.sha256(rendered.encode()).hexdigest())
        token_counts.append(len(token_ids))
    return {
        "chat_template_sha256": rendered_digests,
        "chat_token_counts": token_counts,
    }


def write_table_atomic(destination: Path, table: pa.Table) -> None:
    """Write one parquet output without exposing partial results."""
    partial = destination.with_name(f".{destination.name}.partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        pq.write_table(table, partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def prepare_smoke_split(
    source: Path,
    train_destination: Path,
    val_destination: Path,
    *,
    expected_source_sha256: str,
    expected_source_rows: int,
    train_rows: int,
    val_rows: int,
    tokenizer: Any,
    scorer: Callable[..., Any] = default_compute_score,
) -> dict[str, Any]:
    """Create disjoint fixed train/validation rows and validate their contract."""
    if min(expected_source_rows, train_rows, val_rows) <= 0:
        raise ValueError("row counts must be positive")
    source = source.resolve()
    train_destination = train_destination.resolve()
    val_destination = val_destination.resolve()
    if train_destination == val_destination:
        raise ValueError("train and validation outputs must differ")
    source_sha256 = file_sha256(source)
    if source_sha256 != expected_source_sha256:
        raise ValueError("source SHA256 does not match the frozen dataset")
    table = pq.read_table(source)
    if table.num_rows != expected_source_rows:
        raise ValueError(f"expected {expected_source_rows} rows, got {table.num_rows}")
    rows = table.slice(0, train_rows + val_rows).to_pylist()
    if len(rows) != train_rows + val_rows:
        raise ValueError("source has too few rows for the requested smoke split")
    identifiers = [row_identifier(row) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("smoke rows must have distinct extra_info.index values")
    contract = validate_chat_and_reward(rows, tokenizer, scorer)
    train_table = table.slice(0, train_rows)
    val_table = table.slice(train_rows, val_rows)
    created: list[Path] = []
    try:
        write_table_atomic(train_destination, train_table)
        created.append(train_destination)
        write_table_atomic(val_destination, val_table)
        created.append(val_destination)
    except Exception:
        for destination in created:
            destination.unlink(missing_ok=True)
        raise
    return {
        "status": "PASS",
        "source_sha256": source_sha256,
        "source_rows": table.num_rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_sha256": file_sha256(train_destination),
        "val_sha256": file_sha256(val_destination),
        "train_problem_ids": identifiers[:train_rows],
        "val_problem_ids": identifiers[train_rows:],
        **contract,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-destination", type=Path, required=True)
    parser.add_argument("--val-destination", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-rows", type=int, required=True)
    parser.add_argument("--train-rows", type=int, default=32)
    parser.add_argument("--val-rows", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Run the reproducible smoke-split preparation command."""
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(
        json.dumps(
            prepare_smoke_split(
                args.source,
                args.train_destination,
                args.val_destination,
                expected_source_sha256=args.expected_source_sha256,
                expected_source_rows=args.expected_source_rows,
                train_rows=args.train_rows,
                val_rows=args.val_rows,
                tokenizer=tokenizer,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
