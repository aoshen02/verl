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

"""CPU tests for the ModelOpt MIXED_PRECISION weight rules and their exporter:
the exporter must emit exactly what ``requantize`` produces (the QAT forward
uses the same function) plus the checkpoint's static activation scales."""

import json

import torch
from safetensors.torch import save_file

from verl.utils.modelopt.mixed_precision_export import ModelOptMixedPrecisionExporter, load_mixed_precision_config
from verl.utils.modelopt.mixed_precision_quant import dequantize, nvfp4_decode_values, requantize


def test_fp8_requantize_is_per_tensor_amax_over_448_and_idempotent():
    weight = torch.randn(8, 32, dtype=torch.bfloat16)
    first = requantize("FP8", weight)
    assert first["weight"].dtype == torch.float8_e4m3fn
    assert torch.equal(first["weight_scale"], weight.float().abs().amax() / 448.0)
    deployed = dequantize("FP8", first)
    assert (deployed - weight.float()).abs().max() <= 0.07 * weight.float().abs().amax()
    again = requantize("FP8", weight)
    assert torch.equal(again["weight"].float(), first["weight"].float())


def test_nvfp4_decode_reads_low_nibble_first_with_sign_bit():
    # 0x9F: low nibble 0xF = -6.0, high nibble 0x9 = -0.5; 0x21: low 1 = 0.5, high 2 = 1.0
    packed = torch.tensor([[0x9F, 0x21]], dtype=torch.uint8)
    assert nvfp4_decode_values(packed).tolist() == [[-6.0, -0.5, 0.5, 1.0]]


def test_exporter_requantizes_listed_layers_and_forwards_static_scales(tmp_path):
    quant = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"layers.0.proj": {"quant_algo": "FP8"}},
        }
    }
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant))
    save_file({"layers.0.proj.input_scale": torch.tensor(0.25)}, str(tmp_path / "model.safetensors"))
    index = {
        "weight_map": {"layers.0.proj.input_scale": "model.safetensors", "layers.0.proj.weight": "model.safetensors"}
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    exporter = ModelOptMixedPrecisionExporter(str(tmp_path), load_mixed_precision_config(str(tmp_path)))
    weight = torch.randn(4, 16, dtype=torch.bfloat16)
    other = torch.randn(4, 16, dtype=torch.bfloat16)
    out = dict(exporter([("layers.0.proj.weight", weight), ("layers.1.proj.weight", other)]))

    expected = requantize("FP8", weight)
    assert torch.equal(out["layers.0.proj.weight"].float(), expected["weight"].float())
    assert torch.equal(out["layers.0.proj.weight_scale"], expected["weight_scale"])
    assert out["layers.0.proj.input_scale"].item() == 0.25
    assert out["layers.1.proj.weight"] is other
    assert set(out) == {
        "layers.0.proj.weight",
        "layers.0.proj.weight_scale",
        "layers.0.proj.input_scale",
        "layers.1.proj.weight",
    }
