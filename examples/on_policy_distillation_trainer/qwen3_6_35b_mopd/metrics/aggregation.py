# Copyright 2026 The verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only aggregation. Throughput uses batch makespan, never median latency."""

from collections import defaultdict
from statistics import median


def rate(tokens, seconds, gpus):
    if seconds <= 0 or gpus <= 0:
        raise ValueError("Throughput requires positive duration and GPU count")
    return tokens / seconds / gpus


def summarize_batches(batches, requests, accepted_keys, rollout_gpus, teacher_gpus, flop_fn, peak_tflops):
    """Return scalar metrics and auditable logical-batch records.

    All timestamps are assigned by ONE collector's monotonic clock. flop_fn
    returns training FLOPs; forward-only teacher estimates divide by three,
    matching verl's EngineWorker convention. Failed pre-generation requests
    have unknown token counts and are counted separately, never fabricated.
    """
    accepted = set(accepted_keys)
    known_keys = {row["key"] for row in requests.values()}
    if not accepted <= known_keys:
        raise ValueError("Accepted training keys are missing from the telemetry ledger")
    scalars = defaultdict(list)
    records = []
    for batch in batches:
        rows = [requests[uid] for uid in batch["uids"]]
        if any(r.get("status") not in {"ready", "failure"} for r in rows):
            raise ValueError("Cannot summarize a batch with unfinished requests")
        generated = [r for r in rows if "generated_at" in r]
        if not generated:
            raise ValueError("No observable generated tokens in rollout batch")
        kept = [r for r in generated if r["key"] in accepted]
        generation_s = max(r["generated_at"] for r in generated) - batch["start"]
        ready_s = max(r["end"] for r in rows) - batch["start"]
        total_output = sum(r["output_tokens"] for r in generated)
        effective_output = sum(r["loss_tokens"] for r in kept)
        b = {
            "batch_id": batch["batch_id"],
            "submitted": len(rows),
            "generated": len(generated),
            "accepted": len(kept),
            "failures": sum(r["status"] == "failure" for r in rows),
            "unknown_output_requests": len(rows) - len(generated),
            "generation_s": generation_s,
            "ready_s": ready_s,
            "input_tokens": sum(r["input_tokens"] for r in generated),
            "output_tokens": total_output,
            "effective_output_tokens": effective_output,
        }
        b["filtered_ratio"] = 1 - len(kept) / len(generated)
        b["failure_ratio"] = b["failures"] / len(rows)
        b["output_tok_s_gpu"] = rate(total_output, generation_s, rollout_gpus)
        b["effective_output_tok_s_gpu"] = rate(effective_output, generation_s, rollout_gpus)
        b["ready_output_tok_s_gpu"] = rate(effective_output, ready_s, rollout_gpus)
        for name in (
            "generation_s",
            "ready_s",
            "filtered_ratio",
            "failure_ratio",
            "output_tok_s_gpu",
            "effective_output_tok_s_gpu",
            "ready_output_tok_s_gpu",
        ):
            scalars[f"rollout/batch/{name}_p50"].append(b[name])
        for name in ("input_tokens", "output_tokens", "generation_latency_s", "e2e_latency_s"):
            values = [r[name] for r in generated if name in r]
            if values:
                b[f"request_{name}_p50"] = median(values)
                scalars[f"rollout/request/{name}_p50"].extend(values)
        b["teachers"] = {}
        for domain, gpus in teacher_gpus.items():
            teacher_rows = [r for r in rows if r.get("teacher") == domain and "teacher_end" in r]
            if not teacher_rows:
                continue
            elapsed = max(r["teacher_end"] for r in teacher_rows) - min(r["teacher_start"] for r in teacher_rows)
            lengths = [r["input_tokens"] + r["output_tokens"] for r in teacher_rows]
            t = {
                "requests": len(teacher_rows),
                "window_s": elapsed,
                "scored_tokens": sum(lengths),
                "scored_tok_s_gpu": rate(sum(lengths), elapsed, gpus),
                "mfu_estimate": flop_fn(lengths) / 3 / (elapsed * gpus * peak_tflops * 1e12),
                "request_latency_s_p50": median(r["teacher_latency_s"] for r in teacher_rows),
            }
            b["teachers"][domain] = t
            for name in ("window_s", "scored_tok_s_gpu", "mfu_estimate"):
                scalars[f"teacher/{domain}/batch/{name}_p50"].append(t[name])
            scalars[f"teacher/{domain}/request/latency_s_p50"].extend(r["teacher_latency_s"] for r in teacher_rows)
        records.append(b)
    result = {key: median(values) for key, values in scalars.items()}
    result["rollout/logical_batches"] = len(records)
    result["rollout/submitted"] = sum(b["submitted"] for b in records)
    result["rollout/accepted"] = sum(b["accepted"] for b in records)
    result["rollout/failures"] = sum(b["failures"] for b in records)
    result["rollout/unknown_output_requests"] = sum(b["unknown_output_requests"] for b in records)
    return result, records


def add_teacher_execution(records, execution, step):
    """Attach execution and residual wait/overhead time; fail on missing timing."""
    grouped = defaultdict(list)
    for event in execution:
        if event["step"] == step:
            grouped[(event["batch_id"], event["teacher"])].append(float(event["active_s"]))
    scalars = defaultdict(list)
    for batch in records:
        for domain, teacher in batch["teachers"].items():
            durations = grouped[(batch["batch_id"], domain)]
            if not durations or any(value <= 0 for value in durations):
                raise ValueError(f"Missing or invalid teacher execution timing: {step}, {domain}")
            active = sum(durations)
            if active > teacher["window_s"]:
                raise ValueError(f"Teacher execution exceeds scoring window: {step}, {domain}")
            teacher.update(active_s=active, wait_s=teacher["window_s"] - active,
                           execution_calls=len(durations))
            for name in ("active_s", "wait_s"):
                scalars[f"teacher/{domain}/batch/{name}_p50"].append(teacher[name])
    return {key: median(values) for key, values in scalars.items()}


class TrainTotals:
    """Time-weighted run metrics; not an unweighted mean of per-step rates."""

    def __init__(self):
        self.seconds = self.tokens = self.flops = 0.0
        self.steps = 0

    def update(self, seconds, tokens, flops, gpus, peak_tflops):
        self.seconds += seconds
        self.tokens += tokens
        self.flops += flops
        self.steps += 1
        return {
            "trainer/train_s_mean": self.seconds / self.steps,
            "trainer/train_tok_s_gpu_run": rate(self.tokens, self.seconds, gpus),
            "trainer/train_mfu_estimate_run": self.flops / (self.seconds * gpus * peak_tflops * 1e12),
        }
