# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""vLLM quantization backend defaults for the rollout.

vLLM's reload (``start_reload`` / ``finish_reload``) restores the checkpoint
layout of every quantization scheme for a refit.
"""

import os
from collections.abc import Mapping
from typing import Any

import torch


def _is_nvfp4_quantization(quantization_config: Mapping[str, Any]) -> bool:
    """Whether a quantization config contains the NVFP4 format."""
    values = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list | tuple):
            for nested in value:
                collect(nested)
        elif isinstance(value, str):
            values.append(value.lower())

    collect(quantization_config)
    return any("nvfp4" in value for value in values)


def _uses_fp32_block_scales(quantization_config: Mapping[str, Any]) -> bool:
    """Whether FP8 block weights use checkpoint-layout FP32 scales."""
    for group in quantization_config.get("config_groups", {}).values():
        weights = group.get("weights", {})
        if weights.get("strategy") == "block" and weights.get("scale_dtype") in {None, "torch.float32"}:
            return True
    block_size = quantization_config.get("weight_block_size")
    return block_size == [128, 128] and quantization_config.get("scale_fmt") not in {"ue8m0", "e8m0"}


def _has_mxfp4_weights(quantization_config: Mapping[str, Any]) -> bool:
    """Whether a compressed-tensors config has MXFP4 (4-bit float, group 32) weights."""
    for group in quantization_config.get("config_groups", {}).values():
        weights = group.get("weights") or {}
        if weights.get("num_bits") == 4 and weights.get("type") == "float" and weights.get("group_size") == 32:
            return True
    return False


def _is_hopper() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_device_capability(90)


def set_default_vllm_quantization_backends(
    quantization_config: Mapping[str, Any] | None,
    engine_kwargs: dict[str, Any],
) -> None:
    """Default non-NVFP4 quantization backends to DeepGEMM."""
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    if not isinstance(quantization_config, Mapping) or _is_nvfp4_quantization(quantization_config):
        return
    # DeepGEMM is NVIDIA-only; on ROCm, vLLM's automatic selection picks the AITER / Triton kernels.
    if torch.version.hip is not None:
        return
    # MXFP4 experts take MXFP8 activations, which DeepGEMM does not run
    options = ("linear_backend",) if _has_mxfp4_weights(quantization_config) else ("linear_backend", "moe_backend")
    for option in options:
        engine_kwargs.setdefault(option, "deep_gemm")
    if (
        "deep_gemm" in (engine_kwargs["linear_backend"], engine_kwargs.get("moe_backend"))
        and _is_hopper()
        and _uses_fp32_block_scales(quantization_config)
    ):
        os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"
