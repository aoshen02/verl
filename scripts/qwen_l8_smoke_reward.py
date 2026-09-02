"""Deterministic nonzero training signal for the truncated-model smoke test."""

import hashlib


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, float]:
    del data_source, ground_truth, extra_info, kwargs
    digest = hashlib.sha256(solution_str.encode()).digest()
    score = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    return {"score": score, "smoke_signal": score}
