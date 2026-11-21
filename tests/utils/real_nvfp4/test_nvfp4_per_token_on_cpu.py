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

"""Per-token NVFP4 rollout config, trainer-side export and rollout checks, without vLLM."""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("compressed_tensors")

from verl.utils.qat.compressed_tensors import (  # noqa: E402
    NVFP4PerTokenWeightQuantizer,
    build_compressed_tensors_config,
    dispatch_rollout_weight_quantizer,
    is_nvfp4_per_token_config,
    nvfp4_per_token_module_matcher,
    qat_modes_for_config,
)

_CARVE_IGNORE = [r"re:.*layers\.(0|1|6|7)\.mlp\.experts\..*"]
_HF = SimpleNamespace(num_hidden_layers=8, num_experts=2)


def _config(ignore=_CARVE_IGNORE):
    return build_compressed_tensors_config(
        linear=None, moe="NVFP4", moe_activation="nvfp4_per_token", ignore=ignore, config_file=None
    )


def _dispatch(ignore=_CARVE_IGNORE, activation="nvfp4_per_token", start=2, end=2, engine="vllm", hf_config=_HF):
    quantization = SimpleNamespace(linear=None, moe="NVFP4", moe_activation=activation, ignore=ignore, config_file=None)
    return dispatch_rollout_weight_quantizer(
        model_config=SimpleNamespace(hf_config=hf_config, local_path=None),
        rollout_config=SimpleNamespace(name=engine, quantization=quantization),
        real_nvfp4_bf16_layers=(start, end),
    )


def test_per_token_is_nvfp4_weights_with_dynamic_inputs_and_moe_only():
    group = _config()["config_groups"]["group_moe"]
    assert group["weights"]["strategy"] == "tensor_group" and group["weights"]["group_size"] == 16
    assert group["input_activations"]["dynamic"] is True
    assert is_nvfp4_per_token_config(_config())
    # NVFP4 has no static activation scales, so its activations default to per token.
    assert (
        build_compressed_tensors_config(linear=None, moe="NVFP4", ignore=_CARVE_IGNORE, config_file=None) == _config()
    )
    # The activation override needs moe=NVFP4.
    for moe in (None, "MXFP4"):
        with pytest.raises(ValueError, match="moe_activation='nvfp4_per_token'"):
            build_compressed_tensors_config(
                linear="FP8_BLOCK", moe=moe, moe_activation="nvfp4_per_token", ignore=[], config_file=None
            )
    with pytest.raises(ValueError, match="moe_activation='nvfp4_static'"):
        build_compressed_tensors_config(
            linear=None, moe="NVFP4", moe_activation="nvfp4_static", ignore=[], config_file=None
        )
    with pytest.raises(ValueError, match="NVFP4_PER_TOKEN"):
        build_compressed_tensors_config(linear=None, moe="NVFP4_PER_TOKEN", ignore=[], config_file=None)


def test_per_token_matcher_follows_the_ignore_list_for_any_expert_naming():
    per_token = nvfp4_per_token_module_matcher(_config())
    assert per_token("model.layers.2.mlp.experts.0.gate_proj")
    assert per_token("language_model.model.layers.5.mlp.experts.3.down_proj")
    assert not per_token("model.layers.1.mlp.experts.0.gate_proj")
    assert not per_token("model.layers.2.mlp.shared_experts.gate_proj")
    assert not per_token("model.layers.2.self_attn.q_proj")


def _fake_quantize(weight, global_scale):
    rows, cols = weight.shape
    packed = torch.full((rows, cols // 2), int(global_scale.item()) % 256, dtype=torch.uint8)
    return packed, torch.ones((rows, cols // 16), dtype=torch.float8_e4m3fn)


def _export_experts(quantizer, layers, prefix="model.layers"):
    return list(quantizer([(f"{prefix}.{layer}.mlp.experts.gate_up_proj", torch.ones(2, 8, 32)) for layer in layers]))


@pytest.mark.parametrize(
    "hf_config",
    [
        # DeepSeek-V3 / GLM / Kimi style: leading dense layers and shared experts.
        SimpleNamespace(num_hidden_layers=8, n_routed_experts=4, first_k_dense_replace=1, n_shared_experts=1),
        # Qwen3.5 style: the decoder config is nested under text_config.
        SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=8, num_experts=4)),
    ],
)
def test_export_checks_the_carveout_on_other_layouts(monkeypatch, hf_config):
    monkeypatch.setattr(NVFP4PerTokenWeightQuantizer, "_quantize", staticmethod(_fake_quantize))
    _export_experts(_dispatch(hf_config=hf_config), range(8))
    # Training keeps layers 0, 1 and 7 in BF16, the rollout also layer 6.
    with pytest.raises(ValueError, match=r"layers\.6\.mlp\.experts\.gate_up_proj in BF16"):
        _export_experts(_dispatch(hf_config=hf_config, end=1), range(8))


def test_export_checks_the_carveout_under_the_exported_names(monkeypatch):
    monkeypatch.setattr(NVFP4PerTokenWeightQuantizer, "_quantize", staticmethod(_fake_quantize))
    # An ignore list anchored to a VLM's exported names, not to ``model.layers``.
    ignore = [r"re:model\.language_model\.layers\.(0|1|6|7)\.mlp\.experts\..*"]
    _export_experts(_dispatch(ignore=ignore), range(8), prefix="model.language_model.layers")
    with pytest.raises(ValueError, match=r"layers\.1\.mlp\.experts\.gate_up_proj in BF16"):
        _export_experts(_dispatch(ignore=ignore, start=1), range(8), prefix="model.language_model.layers")
    with pytest.raises(ValueError, match="in NVFP4"):
        _export_experts(_dispatch(ignore=ignore, start=2), range(8), prefix="model.layers")


def test_dispatch_pairs_real_nvfp4_with_per_token_activations():
    assert isinstance(_dispatch(), NVFP4PerTokenWeightQuantizer)
    assert isinstance(_dispatch(activation=None), NVFP4PerTokenWeightQuantizer)
    with pytest.raises(ValueError, match="only with the vLLM rollout"):
        _dispatch(engine="sglang")


def test_dispatch_pairs_nvfp4_qat_with_per_token_activations(tmp_path):
    quantization = SimpleNamespace(
        linear=None, moe="NVFP4", moe_activation=None, ignore=_CARVE_IGNORE, config_file=None
    )
    quantizer = dispatch_rollout_weight_quantizer(
        model_config=SimpleNamespace(hf_config=_HF, local_path=None),
        rollout_config=SimpleNamespace(name="vllm", quantization=quantization),
        qat_config=SimpleNamespace(enable=True, mode="nvfp4"),
    )
    assert isinstance(quantizer, NVFP4PerTokenWeightQuantizer) and quantizer.bf16_expert_layers is None
    assert qat_modes_for_config(_config()) == {"nvfp4"}

    # A static NVFP4 config group is rejected wherever it comes from.
    static = _config()
    static["config_groups"]["group_moe"]["input_activations"].update(dynamic="local", observer="minmax")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(static))
    with pytest.raises(ValueError, match="NVFP4 is supported only with moe_activation=nvfp4_per_token"):
        build_compressed_tensors_config(linear=None, moe=None, ignore=[], config_file=str(config_file))
    native = SimpleNamespace(num_hidden_layers=8, num_experts=2, quantization_config=static)
    with pytest.raises(ValueError, match="NVFP4 is supported only with moe_activation=nvfp4_per_token"):
        dispatch_rollout_weight_quantizer(
            model_config=SimpleNamespace(hf_config=native, local_path=None),
            rollout_config=SimpleNamespace(name="vllm", quantization=None),
            qat_config=SimpleNamespace(enable=True, mode="nvfp4"),
        )


def test_exporter_shares_one_global_scale_across_gate_and_up(monkeypatch):
    monkeypatch.setattr(NVFP4PerTokenWeightQuantizer, "_quantize", staticmethod(_fake_quantize))
    prefix = "model.layers.2.mlp.experts.0"
    gate, up, down = torch.ones(4, 32), torch.full((4, 32), 4.0), torch.ones(32, 4 * 4)
    stream = [
        (f"{prefix}.gate_proj.weight", gate),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", gate),  # carved out: BF16 passthrough
        (f"{prefix}.down_proj.weight", down),
        (f"{prefix}.up_proj.weight", up),
    ]
    out = dict(NVFP4PerTokenWeightQuantizer(_config())(stream))
    assert torch.equal(out["model.layers.0.mlp.experts.0.gate_proj.weight"], gate)
    gate_scale, up_scale = out[f"{prefix}.gate_proj.weight_global_scale"], out[f"{prefix}.up_proj.weight_global_scale"]
    torch.testing.assert_close(gate_scale, torch.tensor([6.0 * 448.0 / 4.0]))
    assert torch.equal(gate_scale, up_scale)
    assert torch.equal(out[f"{prefix}.down_proj.input_global_scale"], torch.ones(1))
    assert out[f"{prefix}.up_proj.weight_packed"].shape == (4, 16)


def test_exporter_unpacks_only_quantized_packed_experts(monkeypatch):
    monkeypatch.setattr(NVFP4PerTokenWeightQuantizer, "_quantize", staticmethod(_fake_quantize))
    gate_up = torch.ones(2, 8, 32)
    stream = [(f"model.layers.{layer}.mlp.experts.gate_up_proj", gate_up) for layer in (0, 2)]
    out = dict(NVFP4PerTokenWeightQuantizer(_config())(stream))
    # The carved-out layer keeps its packed BF16 tensor; the quantized one is split per expert.
    assert torch.equal(out["model.layers.0.mlp.experts.gate_up_proj"], gate_up)
    assert {name for name in out if ".layers.2." in name} == {
        f"model.layers.2.mlp.experts.{expert}.{projection}.{suffix}"
        for expert in range(2)
        for projection in ("gate_proj", "up_proj")
        for suffix in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
    }


def test_exporter_rejects_an_unpaired_gate_proj(monkeypatch):
    monkeypatch.setattr(NVFP4PerTokenWeightQuantizer, "_quantize", staticmethod(_fake_quantize))
    with pytest.raises(RuntimeError, match="missing gate_proj/up_proj"):
        list(
            NVFP4PerTokenWeightQuantizer(_config())(
                [("model.layers.2.mlp.experts.0.gate_proj.weight", torch.ones(4, 32))]
            )
        )


def test_per_token_rollout_check_reads_a_config_file(tmp_path):
    pytest.importorskip("vllm")
    from verl.utils.real_nvfp4 import check_nvfp4_per_token_rollout
    from verl.workers.config.rollout import RolloutQuantizationConfig

    config_file = tmp_path / "quantization_config.json"
    config_file.write_text(json.dumps(_config()))
    built = RolloutQuantizationConfig(config_file=str(config_file)).build_quantization_config(None)
    assert is_nvfp4_per_token_config(built)
    # A checkpoint that ships the config is detected the same way.
    assert is_nvfp4_per_token_config(SimpleNamespace(to_dict=lambda: built))
    static = build_compressed_tensors_config(linear=None, moe="MXFP4", ignore=[], config_file=None)
    assert not is_nvfp4_per_token_config(static)

    rollout = SimpleNamespace(
        tensor_model_parallel_size=4,
        expert_parallel_size=4,
        data_parallel_size=1,
        mtp=None,
        checkpoint_engine=SimpleNamespace(backend="naive"),
    )

    def check(compilation_config, engine_kwargs, config=built):
        check_nvfp4_per_token_rollout(rollout, config, compilation_config, engine_kwargs)

    # Other quantization configs are left alone.
    check_nvfp4_per_token_rollout(rollout, static, {"cudagraph_mode": "FULL_AND_PIECEWISE"}, {"enable_eplb": True})
    # Expert parallelism within one engine keeps whole experts per rank.
    compilation_config = {}
    check(compilation_config, {})
    assert compilation_config == {"cudagraph_mode": "FULL_DECODE_ONLY"}
    with pytest.raises(ValueError, match="FULL_DECODE_ONLY"):
        check({"cudagraph_mode": "FULL_AND_PIECEWISE"}, {})
    with pytest.raises(ValueError, match="no EPLB"):
        check({}, {"enable_eplb": True})
    for context_parallel in ("decode_context_parallel_size", "prefill_context_parallel_size"):
        with pytest.raises(ValueError, match="context parallelism"):
            check({}, {context_parallel: 2})
    # Tensor-parallel experts
    rollout.expert_parallel_size = 1
    check({}, {})
    # Data-parallel attention with all2all expert parallelism
    rollout.tensor_model_parallel_size, rollout.data_parallel_size = 1, 4
    with pytest.raises(ValueError, match="data_parallel_size=1"):
        check({}, {})


def _per_token_rollout(**overrides):
    rollout = dict(data_parallel_size=1, mtp=None, checkpoint_engine=SimpleNamespace(backend="naive"))
    return SimpleNamespace(**{**rollout, **overrides})


@pytest.mark.parametrize(
    "per_token,engine_kwargs,expected",
    [
        (True, {}, {"moe_backend": "flashinfer_trtllm"}),
        (True, {"moe_backend": "cutlass"}, {"moe_backend": "cutlass"}),
        (False, {}, {}),
    ],
    ids=["defaults-trtllm", "keeps-user-backend", "other-config-untouched"],
)
def test_per_token_rollout_check_defaults_moe_backend(per_token, engine_kwargs, expected):
    from verl.utils.real_nvfp4 import check_nvfp4_per_token_rollout

    config = (
        _config()
        if per_token
        else build_compressed_tensors_config(linear=None, moe="MXFP4", ignore=[], config_file=None)
    )
    check_nvfp4_per_token_rollout(_per_token_rollout(), config, {}, engine_kwargs)
    assert engine_kwargs == expected
