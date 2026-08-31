from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import prepare_dapo_math_rl_smoke as smoke
from scripts.prepare_dapo_math_rl_smoke import file_sha256, prepare_smoke_split


class FakeTokenizer:
    def apply_chat_template(self, prompt, *, tokenize, add_generation_prompt):
        assert not tokenize
        assert add_generation_prompt
        return "\n".join(message["content"] for message in prompt) + "\nassistant:"

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [1] if text else []}


def fake_scorer(data_source, solution, ground_truth):
    assert data_source == "math_dapo"
    return {"score": 1.0 if solution == f"Answer: {ground_truth}" else -1.0}


def write_source(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def row(index: str, answer: str) -> dict:
    return {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": f"solve {index}"}],
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {"index": index},
    }


def test_prepare_smoke_split_writes_disjoint_validated_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    write_source(source, [row("a", "1"), row("b", "2"), row("c", "3")])

    result = prepare_smoke_split(
        source,
        tmp_path / "train.parquet",
        tmp_path / "val.parquet",
        expected_source_sha256=file_sha256(source),
        expected_source_rows=3,
        train_rows=2,
        val_rows=1,
        tokenizer=FakeTokenizer(),
        scorer=fake_scorer,
    )

    assert result["status"] == "PASS"
    assert result["train_problem_ids"] == ["a", "b"]
    assert result["val_problem_ids"] == ["c"]
    assert pq.read_table(tmp_path / "train.parquet").num_rows == 2
    assert pq.read_table(tmp_path / "val.parquet").num_rows == 1


def test_prepare_smoke_split_rejects_duplicate_ids_without_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    write_source(source, [row("a", "1"), row("a", "2")])
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"

    with pytest.raises(ValueError, match="distinct"):
        prepare_smoke_split(
            source,
            train,
            val,
            expected_source_sha256=file_sha256(source),
            expected_source_rows=2,
            train_rows=1,
            val_rows=1,
            tokenizer=FakeTokenizer(),
            scorer=fake_scorer,
        )

    assert not train.exists()
    assert not val.exists()


def test_real_dapo_reward_accepts_answer_and_rejects_wrong_answer() -> None:
    assert smoke.default_compute_score("math_dapo", "Answer: 5", "5")["score"] == 1.0
    assert smoke.default_compute_score("math_dapo", "Answer: 6", "5")["score"] == -1.0


def test_prepare_smoke_split_removes_train_output_if_val_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.parquet"
    write_source(source, [row("a", "1"), row("b", "2")])
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    original_write = smoke.write_table_atomic
    calls = 0

    def fail_on_val(destination: Path, table: pa.Table) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated val write failure")
        original_write(destination, table)

    monkeypatch.setattr(smoke, "write_table_atomic", fail_on_val)
    with pytest.raises(OSError, match="simulated"):
        prepare_smoke_split(
            source,
            train,
            val,
            expected_source_sha256=file_sha256(source),
            expected_source_rows=2,
            train_rows=1,
            val_rows=1,
            tokenizer=FakeTokenizer(),
            scorer=fake_scorer,
        )

    assert not train.exists()
    assert not val.exists()
