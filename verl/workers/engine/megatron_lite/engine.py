# Copyright 2026 aoshen02
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

"""Run Nemotron through verl's normal engine and checkpoint transport APIs."""

from contextlib import contextmanager
from dataclasses import fields

import torch
import torch.distributed as dist
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts import OptimizerConfig
from megatron.lite.runtime.contracts.loss import LossContext
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.v1.worker.workspace import init_workspace_manager

from verl.utils import tensordict_utils as tu
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches

from .data import pack_actor_batch, unpack_logprobs


@EngineRegistry.register("language_model", "megatron_lite")
class MegatronLiteEngine(BaseEngine):
    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        path = model_config.local_path or model_config.path
        self.runtime_config = MegatronLiteConfig.from_dict(path, engine_config.runtime_config)
        parallel = self.runtime_config.parallel
        if parallel.tp != 1 or parallel.cp != 1 or parallel.vpp != 1:
            raise NotImplementedError("Aligned Nemotron engine currently requires TP1/CP1/VPP1")
        if self.runtime_config.model_name != "nemotron_h":
            raise ValueError("This engine is restricted to the reviewed Nemotron recipe")
        if not dist.is_initialized():
            raise RuntimeError("TrainingWorker must initialize the distributed environment")
        world, rank = dist.get_world_size(), dist.get_rank()
        if world % parallel.pp:
            raise ValueError("World size is not divisible by pipeline size")
        self.dp_size = world // parallel.pp
        self.dp_rank = rank % self.dp_size
        self.output_rank = rank // self.dp_size == parallel.pp - 1
        names = {f.name for f in fields(OptimizerConfig)}
        opt = {name: getattr(optimizer_config, name) for name in names if hasattr(optimizer_config, name)}
        opt["adam_beta1"], opt["adam_beta2"] = optimizer_config.betas
        self.runtime_config.optimizer = OptimizerConfig(**opt)
        overrides = getattr(optimizer_config, "override_optimizer_config", None)
        if overrides:
            self.runtime_config.optimizer.override_optimizer_config = dict(overrides)
        self.runtime = MegatronLiteRuntime(path, self.runtime_config)
        self.handle = None
        self.mode = None
        self.vllm_config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["none", "+quant_fp8"]))
        self.vllm_config.kernel_config.moe_backend = "humming"

    def initialize(self):
        if self.handle is not None:
            raise RuntimeError("Repeated reset requires an explicit checkpoint restore")
        torch.set_default_dtype(torch.bfloat16)
        with set_current_vllm_config(self.vllm_config):
            init_distributed_environment(world_size=dist.get_world_size(), rank=dist.get_rank())
            initialize_model_parallel(1, self.runtime_config.parallel.pp)
            init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
            self.handle = self.runtime.build_model()
        if (self.handle.dp_size, self.handle.dp_rank) != (self.dp_size, self.dp_rank):
            raise RuntimeError("Runtime and verl data-parallel decomposition differ")

    @property
    def is_param_offload_enabled(self):
        return self.engine_config.param_offload

    @property
    def is_optimizer_offload_enabled(self):
        return self.engine_config.optimizer_offload

    @contextmanager
    def _mode(self, mode, **kwargs):
        if self.handle is None:
            raise RuntimeError("Engine is not initialized")
        runtime_mode = self.runtime.train_mode if mode == "train" else self.runtime.eval_mode
        with BaseEngineCtx(self, mode, **kwargs), runtime_mode(self.handle):
            yield

    def train_mode(self, **kwargs):
        return self._mode("train", **kwargs)

    def eval_mode(self, **kwargs):
        return self._mode("eval", **kwargs)

    def optimizer_zero_grad(self):
        self.runtime.zero_grad(self.handle)

    def optimizer_step(self):
        success, norm, _ = self.runtime.optimizer_step(self.handle)
        if not success:
            raise RuntimeError("Optimizer skipped the policy update")
        return norm

    def lr_scheduler_step(self):
        return self.runtime.lr_scheduler_step(self.handle)

    def forward_backward_batch(self, data, loss_function, forward_only=False):
        if self.handle is None:
            raise RuntimeError("Engine is not initialized")
        if not forward_only and loss_function is None:
            raise ValueError("Actor training requires the framework loss function")
        group = self.get_data_parallel_group()
        mask = data["response_mask"]
        count = mask.values().sum().float()
        dist.all_reduce(count, group=group)
        tu.assign_non_tensor(data, batch_num_tokens=count.item(), dp_size=self.dp_size, sp_size=1)
        micro_batches, indices = prepare_micro_batches(data, dp_group=group)
        packed = []
        for micro in micro_batches:
            temperature = torch.as_tensor(tu.get(micro, "temperature", default=1.0)).flatten()
            if not temperature.numel() or not torch.isfinite(temperature).all():
                raise ValueError("Invalid actor temperature")
            if not torch.all(temperature == temperature[0]) or temperature[0] <= 0:
                raise ValueError("Nemotron requires a uniform positive microbatch temperature")
            context = LossContext(
                temperature=float(temperature[0]),
                calculate_entropy=tu.get_non_tensor_data(micro, "calculate_entropy", default=False),
                source_batch=micro,
            )
            packed.append((pack_actor_batch(micro), context))
        results = []

        def loss_callback(output, batch, context):
            model_output = {
                key: unpack_logprobs(output[key], batch) for key in ("log_probs", "entropy") if key in output
            }
            if loss_function is None:
                loss = output["loss"] * 0
                metrics = {}
            else:
                loss, metrics = loss_function(model_output=model_output, data=context.source_batch, dp_group=group)
            results.append(
                {
                    "model_output": {k: v.detach() for k, v in model_output.items()},
                    "loss": float(loss.detach()),
                    "metrics": metrics,
                }
            )
            # The runtime divides by the number of microbatches. verl's loss
            # already normalizes by the global batch, so cancel that division.
            return loss * len(packed), metrics

        with set_current_vllm_config(self.vllm_config):
            self.runtime.forward_backward(
                self.handle,
                packed,
                loss_callback,
                num_microbatches=len(packed),
                forward_only=forward_only,
            )
        return postprocess_batch_func(results, indices, data)

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False):
        if self.handle is None:
            raise RuntimeError("Engine is not initialized")
        if layered_summon:
            raise NotImplementedError("Layered weight export is not integrated")

        def weights():
            with set_current_vllm_config(self.vllm_config):
                yield from self.runtime.export_weights(self.handle)

        return weights(), None

    def get_data_parallel_size(self):
        return self.dp_size

    def get_data_parallel_rank(self):
        return self.dp_rank

    def get_data_parallel_group(self):
        if self.handle is None:
            raise RuntimeError("Data-parallel group is unavailable before initialization")
        return self.handle.dp_group

    def is_mp_src_rank_with_outputs(self):
        return self.output_rank

    def to(self, device, model=True, optimizer=True, grad=True):
        self.runtime.to(self.handle, device, model=model, optimizer=optimizer, grad=grad)

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        if hdfs_path or max_ckpt_to_keep:
            raise NotImplementedError("Remote checkpoint storage/retention is not integrated")
        self.runtime.save_checkpoint(self.handle, local_path, global_step=global_step, **kwargs)

    def load_checkpoint(self, local_path, hdfs_path=None, **kwargs):
        if hdfs_path:
            raise NotImplementedError("Remote checkpoint storage is not integrated")
        return self.runtime.load_checkpoint(self.handle, local_path, **kwargs)

    def close(self):
        if self.handle is not None:
            self.runtime.close(self.handle)
            self.handle = None
