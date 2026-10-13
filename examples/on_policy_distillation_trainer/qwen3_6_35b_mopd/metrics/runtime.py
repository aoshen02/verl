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
"""Instrumentation through example-local V1 subclasses; no upstream loss patches."""

import hashlib
import json
import time
from pathlib import Path

import hydra
import ray
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.v1 import AgentLoopManagerTQ, AgentLoopWorkerTQ, PPOTrainerSync
from verl.utils.flops_counter import _estimate_qwen3_5_flops
from verl.utils.tensordict_utils import get_non_tensor_data

from .aggregation import TrainTotals, add_teacher_execution, rate, summarize_batches

COLLECTOR = "qwen3_6_mopd_telemetry"


def validate_teacher_loading(config):
    """Teachers have no actor weight-sync source, so they must load checkpoints."""
    from verl.utils.config import omega_conf_to_dataclass

    distillation = omega_conf_to_dataclass(config.distillation)
    for domain, teacher in distillation.teacher_models.items():
        if (teacher.num_replicas != 1 or teacher.inference.pipeline_model_parallel_size != 1
                or teacher.inference.data_parallel_size != 1):
            raise ValueError("Teacher execution timing requires one replica and pipeline/data parallel size 1")
        if teacher.inference.load_format not in {"auto", "safetensors"}:
            raise ValueError(f"Teacher {domain} must load checkpoint weights, got {teacher.inference.load_format!r}")


def train_flops(config, lengths):
    if config.model_type != "qwen3_5_moe":
        raise ValueError("This example's FLOP estimator requires qwen3_5_moe")
    return _estimate_qwen3_5_flops(config, sum(lengths), lengths, 1.0) * 1e12


@ray.remote(num_cpus=0)
class Collector:
    """One clock and one writer for all nodes; a few RPCs per long request."""

    def __init__(self, output_dir):
        self.batches = []
        self.requests = {}
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = (self.output / "events.jsonl").open("x", buffering=1)

    def _write(self, event):
        self.events.write(json.dumps(event) + "\n")

    def submit(self, step, uids):
        if any(batch["step"] == step for batch in self.batches):
            raise ValueError("Teacher execution timing requires one synchronous logical batch per step")
        batch = {"step": step, "batch_id": len(self.batches), "uids": uids, "start": time.perf_counter()}
        self.batches.append(batch)
        marker = self.output / "teacher-timing-step.json"
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({"step": step, "batch_id": batch["batch_id"]}))
        temporary.replace(marker)
        for uid in uids:
            if uid in self.requests:
                raise ValueError("Duplicate rollout uid")
            self.requests[uid] = {"uid": uid, "key": f"{uid}_0_0", "status": "pending", "step": step}
        self._write({"event": "submit", **batch})

    def event(self, uid, kind, values=None):
        row = self.requests[uid]
        now = time.perf_counter()
        row.update(values or {})
        if kind == "generated":
            row["generated_at"] = now
        elif kind in {"teacher_start", "teacher_end"}:
            row[kind] = now
        elif kind in {"ready", "failure"}:
            row.update(status=kind, end=now)
        self._write({"event": kind, "at": now, "uid": uid, "step": row["step"], **(values or {})})

    def snapshot(self, step):
        batches = [b for b in self.batches if b["step"] == step]
        return batches, {uid: self.requests[uid] for b in batches for uid in b["uids"]}

    def write_step(self, step, batches):
        with (self.output / "batches.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "batches": batches}) + "\n")


# Upstream exports this worker already decorated with @ray.remote. Subclass its
# Python implementation, then decorate our subclass. This Ray API is internal:
# preflight and the pinned upstream/Ray versions are mandatory for this example.
_TQWorker = AgentLoopWorkerTQ.__ray_metadata__.modified_class


@ray.remote
class MeasuredWorker(_TQWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collector = ray.get_actor(COLLECTOR)
        self.started = {}

    async def _run_agent_loop(self, sampling_params, trajectory, **kwargs):
        uid = kwargs["uid"]
        if trajectory["validate"]:
            return await super()._run_agent_loop(sampling_params, trajectory, **kwargs)
        self.started[uid] = time.perf_counter()
        try:
            return await super()._run_agent_loop(sampling_params, trajectory, **kwargs)
        except Exception as exc:
            await self.collector.event.remote(uid, "failure", {"error_type": type(exc).__name__})
            raise
        finally:
            self.started.pop(uid, None)

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        if validate:
            return await super()._agent_loop_postprocess(output, validate, **kwargs)
        if isinstance(output, list):
            raise ValueError("Measured MOPD supports one single-turn output per request")
        uid = kwargs["uid"]
        await self.collector.event.remote(
            uid,
            "generated",
            {
                "input_tokens": len(output.prompt_ids),
                "output_tokens": len(output.response_ids),
                "loss_tokens": sum(output.response_mask),
                "generation_latency_s": output.metrics.generate_sequences,
            },
        )
        # Upstream computes the diagnostic score, queries the routed teacher,
        # and puts the result in TransferQueue. Record ready before publishing
        # via the teacher hook so the trainer cannot race telemetry collection.
        return await super()._agent_loop_postprocess(output, validate, **kwargs)

    async def _compute_teacher_logprobs(self, output, prompt_ids, response_ids, validate, sample_kwargs=None):
        if validate:
            return await super()._compute_teacher_logprobs(output, prompt_ids, response_ids, validate, sample_kwargs)
        uid = sample_kwargs["uid"]
        domain = str(sample_kwargs["data_source"])
        await self.collector.event.remote(uid, "teacher_start", {"teacher": domain})
        started = time.perf_counter()
        await super()._compute_teacher_logprobs(output, prompt_ids, response_ids, validate, sample_kwargs)
        elapsed = time.perf_counter() - started
        await self.collector.event.remote(uid, "teacher_end", {"teacher_latency_s": elapsed})
        await self.collector.event.remote(uid, "ready", {"e2e_latency_s": time.perf_counter() - self.started[uid]})


class MeasuredAgentLoopManager(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = MeasuredWorker
        self.collector = ray.get_actor(COLLECTOR)

    def generate_sequences(self, prompts):
        # TensorDict.get() preserves NonTensorData wrappers; unwrap metadata
        # before interpreting it as a Python scalar.
        if not get_non_tensor_data(prompts, "validate", False):
            step = int(get_non_tensor_data(prompts, "global_steps", None))
            uids = [str(uid) for uid in prompts["uid"]]
            ray.get(self.collector.submit.remote(step, uids))
        return super().generate_sequences(prompts)


class MeasuredTrainer(PPOTrainerSync):
    def on_init_end(self):
        super().on_init_end()
        self.totals = TrainTotals()
        self.collector = ray.get_actor(COLLECTOR)
        from transformers import AutoConfig

        self.flop_config = AutoConfig.from_pretrained(self.config.actor_rollout_ref.model.path)
        baseline_path = self.config.measurement.teacher_baselines
        self.baselines = json.loads(Path(baseline_path).read_text()) if baseline_path else {"domains": {}}
        from ..prepare_models import MODELS

        if baseline_path and set(self.baselines["domains"]) != {"caesar_cipher", "simple_geometry"}:
            raise ValueError("Both teacher baselines are required")
        for domain, baseline in self.baselines["domains"].items():
            dev = Path(self.config.data.val_files[0]).parent / f"{domain}-dev.parquet"
            if baseline["dev_sha256"] != hashlib.sha256(dev.read_bytes()).hexdigest():
                raise ValueError(f"Teacher and student evaluation data differ: {domain}")
            if baseline["model"] != list(MODELS[domain]):
                raise ValueError(f"Teacher baseline checkpoint differs: {domain}")
            if baseline["decoding"] != {
                "temperature": 1.0,
                "top_p": 1.0,
                "n": 4,
                "max_tokens": 30720,
                "enable_thinking": True,
            }:
                raise ValueError("Teacher baseline decoding differs from the recipe")

    def on_step_begin(self):
        super().on_step_begin()
        self.dispatch_s = self.sample_wait_s = 0.0

    def prepare_step(self):
        start = time.perf_counter()
        result = super().prepare_step()
        self.dispatch_s += time.perf_counter() - start
        return result

    def on_sample_begin(self):
        super().on_sample_begin()
        self.sample_started = time.perf_counter()

    def on_sample_end(self):
        self.sample_wait_s += time.perf_counter() - self.sample_started
        super().on_sample_end()

    def _validate(self):
        metrics = super()._validate()
        for domain in ("caesar_cipher", "simple_geometry"):
            candidates = [v for k, v in metrics.items() if k.endswith(f"/{domain}/acc/mean@4")]
            if len(candidates) != 1:
                raise ValueError(f"Expected one avg@4 accuracy metric for {domain}, found {len(candidates)}")
            metrics[f"eval/student/{domain}/avg_at_4"] = candidates[0]
            references = self.config.measurement.get("teacher_references", {})
            if domain in references:
                metrics[f"eval/teacher_reported/{domain}/avg_at_4"] = references[domain].avg_at_4
                metrics[f"eval/gap/{domain}/student_minus_reported_teacher_avg_at_4"] = (
                    candidates[0] - references[domain].avg_at_4
                )
            if domain in self.baselines["domains"]:
                baseline = self.baselines["domains"][domain]
                metrics[f"eval/teacher/{domain}/accuracy"] = baseline["accuracy"]
                metrics[f"eval/gap/{domain}/student_minus_teacher"] = candidates[0] - baseline["accuracy"]
        return metrics

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        gpus = self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
        peak = self.config.measurement.bf16_peak_tflops_per_gpu
        tags = [tag for tag in batch.tags if not tag.get("is_padding", False)]
        lengths = [tag["seq_len"] for tag in tags]
        train_s = timing_raw["update_actor"]
        flops = train_flops(self.flop_config, lengths) * self.config.actor_rollout_ref.actor.ppo_epochs
        tokens = sum(lengths) * self.config.actor_rollout_ref.actor.ppo_epochs
        wait_s = self.dispatch_s + self.sample_wait_s
        metrics.update(
            {
                "trainer/step_s": timing_raw["step"],
                "trainer/wait_s": wait_s,
                "trainer/train_s": train_s,
                "trainer/other_s": timing_raw["step"] - wait_s - train_s,
                "trainer/eval_s": timing_raw.get("testing", 0.0),
                "trainer/step_plus_eval_s": timing_raw["step"] + timing_raw.get("testing", 0.0),
                "trainer/train_tok_s_gpu": rate(tokens, train_s, gpus),
                "trainer/train_mfu_estimate": flops / (train_s * gpus * peak * 1e12),
            }
        )
        metrics.update(self.totals.update(train_s, tokens, flops, gpus, peak))
        batches, requests = ray.get(self.collector.snapshot.remote(global_steps))
        teacher_gpus = {
            key: model.world_size
            for key, model in self.teacher_model_manager.distillation_config.teacher_models.items()
        }
        accepted_keys = [
            key for key, tag in zip(batch.keys, batch.tags, strict=True) if not tag.get("is_padding", False)
        ]
        scalars, records = summarize_batches(
            batches,
            requests,
            accepted_keys,
            gpus,
            teacher_gpus,
            lambda seqs: train_flops(self.flop_config, seqs),
            peak,
        )
        metrics.update(scalars)
        execution = []
        for domain in teacher_gpus:
            path = Path(self.config.measurement.output_dir) / f"teacher-execution-{domain}.jsonl"
            with path.open() as stream:
                execution.extend(json.loads(line) for line in stream if line.strip())
        metrics.update(add_teacher_execution(records, execution, global_steps))
        ray.get(self.collector.write_step.remote(global_steps, records))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        OmegaConf.resolve(config)
        validate_teacher_loading(config)
        config.transfer_queue.enable = True
        if config.trainer.v1.trainer_mode != "sync" or config.actor_rollout_ref.rollout.n != 1:
            raise ValueError("This example measures synchronous, single-sample MOPD only")
        if config.trainer.resume_mode != "disable":
            raise ValueError("Use a fresh run directory: telemetry resume is not implemented")
        if config.algorithm.rollout_correction.rollout_rs is not None:
            raise ValueError("Token rejection sampling requires accounting for the final actor loss mask")
        evaluation = config.actor_rollout_ref.rollout.val_kwargs
        if (
            evaluation.temperature != 1.0
            or evaluation.top_p != 1.0
            or evaluation.top_k != -1
            or evaluation.n != 4
            or not evaluation.do_sample
        ):
            raise ValueError("Teacher comparison requires sampled avg@4 at temperature=1.0, top_p=1.0, top_k=-1")
        if config.data.max_response_length != 30720 or config.data.max_prompt_length != 2048:
            raise ValueError(
                "Length changes require a separate matched teacher baseline; use the documented smoke plan"
            )
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config

        validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
        if need_critic(config) or need_reference_policy(config):
            raise ValueError("This example uses two frozen teachers with no value critic or reference policy")
        collector = Collector.options(name=COLLECTOR).remote(config.measurement.output_dir)
        tq.init(config.transfer_queue)
        trainer = None
        success = False
        try:
            trainer = MeasuredTrainer(config=config)
            trainer.init()
            manager = MeasuredAgentLoopManager.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
            success = True
        finally:
            if trainer is not None and getattr(trainer, "logger", None):
                trainer.logger.finish(exit_code=0 if success else 1)
            tq.close()
            ray.kill(collector)


@hydra.main(config_path="..", config_name="config", version_base=None)
def main(config):
    run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
