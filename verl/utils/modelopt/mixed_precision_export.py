# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Weight sync into a ModelOpt MIXED_PRECISION deployment (NVFP4 + FP8) from ``qat.mode=modelopt_mixed``.

The trainer holds BF16 masters under ModelOpt's Lightning recipe (static, calibrated weight
quantizers; see ``quantize.apply_mixed_precision_qat``). Every sync quantizes the current weights
with ModelOpt's exporter rule (``to_quantized_weight``) and the static scales of those quantizers,
the same call the QAT forward dequantizes, and sends the quantizers' static activation scales
(``input_scale``, ``k_scale``, ``v_scale``).

Megatron-Bridge yields each weight after its EP gather, so every rank needs the scales of all
experts: they are gathered once over the EP group at the first sync.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator

import torch


def load_mixed_precision_config(model_path: str) -> dict | None:
    path = os.path.join(model_path, "hf_quant_config.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        quant = json.load(f)["quantization"]
    if quant.get("quant_algo") != "MIXED_PRECISION":
        return None
    return quant


def _category(module: str) -> str:
    for name, pattern in (
        ("routed_experts", r"\.experts\.\d+\."),
        ("shared_experts", r"\.shared_experts\."),
        ("mamba_in_proj", r"\.in_proj$"),
        ("mamba_out_proj", r"\.out_proj$"),
    ):
        if re.search(pattern, module):
            return name
    return module


class ModelOptMixedPrecisionExporter:
    def __init__(self, model_path: str, quant: dict):
        from verl.utils.modelopt.quantize import DEPLOY_STATE

        if not DEPLOY_STATE["scales"]:
            raise RuntimeError("mixed_precision_export needs the quantizers of qat.mode=modelopt_mixed")
        self.layers = {name: cfg["quant_algo"] for name, cfg in quant["quantized_layers"].items()}
        self.state = DEPLOY_STATE
        self.scales: dict | None = None
        self.verify = os.environ.get("VERL_QAT_VERIFY_DEPLOYED") == "1"
        self.num_syncs = 0

    def _gather_scales(self) -> dict:
        from megatron.core import parallel_state as mpu

        local = self.state["scales"]
        scales = dict(local)
        if mpu.get_expert_model_parallel_world_size() > 1:
            group = mpu.get_expert_model_parallel_group()
            gathered = [None] * torch.distributed.get_world_size(group=group)
            # float8_e4m3fn tensors do not survive all_gather_object's pickling: send their bytes.
            payload = {
                name: {
                    **entry,
                    "scales": tuple(
                        (str(s.dtype), s.contiguous().reshape(-1).view(torch.uint8)) for s in entry["scales"]
                    ),
                }
                for name, entry in local.items()
                if entry["expert"]
            }
            torch.distributed.all_gather_object(gathered, payload, group=group)
            for rank_scales in gathered:
                for name, entry in rank_scales.items():
                    restored = tuple(
                        raw.view(getattr(torch, dtype.removeprefix("torch."))).reshape(ref.shape)
                        for (dtype, raw), ref in zip(entry["scales"], self._like(local, name), strict=True)
                    )
                    scales[name] = {**entry, "scales": restored}
        self._check_replicas(local)
        missing = set(self.layers) - set(scales)
        if missing or any(scales[name]["algo"] != algo for name, algo in self.layers.items()):
            raise RuntimeError(f"quantizers do not cover the deployment: missing {sorted(missing)[:5]}")
        return scales

    @staticmethod
    def _check_replicas(local: dict) -> None:
        """Every rank calibrates the non-expert layers itself: their scales must agree on all ranks."""
        import hashlib

        def digest(scales):
            h = hashlib.sha256()
            for s in scales:
                h.update(str((s.dtype, tuple(s.shape))).encode())
                h.update(s.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            return h.hexdigest()

        from verl.utils.modelopt.quantize import DEPLOY_STATE

        mine = {name: digest(entry["scales"]) for name, entry in local.items() if not entry["expert"]}
        mine.update(
            {f"{name}.{suffix}": digest((t,)) for name, d in DEPLOY_STATE["static"].items() for suffix, t in d.items()}
        )
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            return
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, mine)
        differing = sorted({name for other in gathered for name in mine if other.get(name) != mine[name]})
        if differing:
            raise RuntimeError(
                f"non-expert quantizer scales differ across ranks: {differing[:8]} ({len(differing)} layers)"
            )
        print(
            f"[QAT-XRANK rank={torch.distributed.get_rank()}] {len(mine)} replicated scale sets identical on "
            f"{len(gathered)} ranks",
            flush=True,
        )

    @staticmethod
    def _like(local: dict, name: str) -> tuple:
        """Scales of a local expert of the same projection: the shapes of every expert's scales."""
        proj = name.rpartition(".")[2]
        return next(e["scales"] for n, e in local.items() if e["expert"] and n.endswith("." + proj))

    def __call__(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        from verl.utils.modelopt.quantize import _dequantize, official_quantize

        if self.scales is None:
            self.scales = self._gather_scales()
        self.num_syncs += 1
        checked: dict[str, list] = {}
        for name, tensor in named_tensors:
            if "_quantizer." in name:
                continue
            module, _, leaf = name.rpartition(".")
            entry = self.scales.get(module) if leaf == "weight" else None
            if entry is None:
                yield name, tensor
            else:
                scales = tuple(s.to(tensor.device) for s in entry["scales"])
                quantized = official_quantize(entry["algo"], tensor, scales)
                if self.verify and module in self.state["local"]:
                    # The weight the QAT forward of this rank uses vs the tensors on the wire.
                    quantizer, weight = self.state["local"][module]
                    with torch.no_grad():
                        forward = quantizer(weight.detach())
                        deployed = _dequantize(entry["algo"], quantized).to(forward.dtype)
                        diff = (forward.float() - deployed.float()).abs()
                    row = checked.setdefault(_category(module), [0, 0, 0, 0.0])
                    row[0] += 1
                    row[1] += diff.numel()
                    row[2] += int((diff != 0).sum())
                    row[3] = max(row[3], float(diff.max()))
                for suffix, value in quantized.items():
                    yield f"{module}.{suffix}", value
            if leaf == "weight":
                for suffix, static in self.state["static"].get(module, {}).items():
                    yield f"{module}.{suffix}", static.to(tensor.device)
        if self.verify:
            from verl.utils.modelopt.quantize import state_digest

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[QAT-STATE rank={rank} sync={self.num_syncs}] {state_digest()}", flush=True)
            for category, (tensors, elements, mismatches, max_abs) in sorted(checked.items()):
                print(
                    f"[QAT-VERIFY rank={rank} sync={self.num_syncs}] {category}: tensors={tensors} "
                    f"elements={elements} forward!=deployed={mismatches} max_abs={max_abs:.3e}",
                    flush=True,
                )
