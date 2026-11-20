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

"""``set_default_vllm_quantization_backends`` engine defaults, without vLLM."""

import importlib.util
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def quant_utils(monkeypatch):
    # Load by path: the verl.utils.vllm package imports vLLM.
    path = _REPO_ROOT / "verl/utils/vllm/vllm_quant_utils.py"
    spec = importlib.util.spec_from_file_location("vllm_quant_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_is_hopper", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None)
    return module


def _ct(weights):
    return {"quant_method": "compressed-tensors", "config_groups": {"group_0": {"weights": weights}}}


_FP8_BLOCK = _ct({"num_bits": 8, "type": "float", "strategy": "block", "block_structure": [128, 128]})
_MXFP4 = _ct({"num_bits": 4, "type": "float", "strategy": "group", "group_size": 32})
_NVFP4 = _ct({"num_bits": 4, "type": "float", "strategy": "tensor_group", "group_size": 16, "format": "nvfp4"})
_DEEP_GEMM = {"linear_backend": "deep_gemm", "moe_backend": "deep_gemm"}


@pytest.mark.parametrize(
    "config,engine_kwargs,expected",
    [
        (_FP8_BLOCK, {}, _DEEP_GEMM),
        (_MXFP4, {}, {"linear_backend": "deep_gemm"}),
        (_MXFP4, {"moe_backend": "marlin"}, {"linear_backend": "deep_gemm", "moe_backend": "marlin"}),
        (_FP8_BLOCK, {"moe_backend": "triton"}, {"linear_backend": "deep_gemm", "moe_backend": "triton"}),
        (_NVFP4, {}, {}),
        (None, {}, {}),
    ],
    ids=["fp8-both-deep-gemm", "mxfp4-moe-unset", "mxfp4-keeps-user-moe", "keeps-user-moe", "nvfp4-untouched", "none"],
)
def test_default_backends(quant_utils, config, engine_kwargs, expected):
    quant_utils.set_default_vllm_quantization_backends(config, engine_kwargs)
    assert engine_kwargs == expected
