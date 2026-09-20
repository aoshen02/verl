# Copyright 2026 individual contributors.
# SPDX-License-Identifier: Apache-2.0
"""Packing must preserve logical delta-W, including independent expert factors."""

import pytest
import torch

from verl.utils.inkling_lora import convert_inkling_lora, pack_dense, pack_shared, unpack_experts


def test_dense_packing_preserves_interleaved_rows_and_scale():
    torch.manual_seed(7)
    ag, au = torch.randn(2, 5, dtype=torch.float64), torch.randn(2, 5, dtype=torch.float64)
    bg, bu = torch.randn(3, 2, dtype=torch.float64), torch.randn(3, 2, dtype=torch.float64)
    a, b = pack_dense((ag, bg), (au, bu))
    expected = torch.stack((bg @ ag, bu @ au), dim=1).flatten(0, 1)
    torch.testing.assert_close(b @ a, expected)
    tensors = {
        f"base_model.model.model.layers.0.mlp.{proj}.lora_{factor}.weight": tensor
        for proj, pair in [("gate_proj", (ag, bg)), ("up_proj", (au, bu))]
        for factor, tensor in zip(("A", "B"), pair, strict=True)
    }
    output, config = convert_inkling_lora(tensors, {"r": 2, "lora_alpha": 6}, num_experts=4, num_shared_experts=2)
    prefix = "model.layers.0.mlp.gate_up_proj"
    actual = output[prefix + ".lora_B.weight"] @ output[prefix + ".lora_A.weight"]
    torch.testing.assert_close(actual * config["lora_alpha"] / config["r"], expected * 3)


@pytest.mark.parametrize("down", [False, True])
def test_shared_packing_preserves_each_expert_delta(down):
    torch.manual_seed(9)
    a = torch.randn(2, 3, 5, dtype=torch.float64)
    b = torch.randn(2, 7, 3, dtype=torch.float64)
    packed_a, packed_b = pack_shared(a, b, down=down)
    deltas = (b @ a).unbind()
    expected = torch.cat(deltas, dim=1 if down else 0)
    torch.testing.assert_close(packed_b @ packed_a, expected)


def test_peft_expert_axes_and_missing_pair():
    a = torch.arange(30).reshape(3, 2, 5)
    b = torch.arange(42).reshape(3, 7, 2)
    actual_a, actual_b = unpack_experts(a.flatten(0, 1), b.permute(1, 2, 0).flatten(1), 3)
    torch.testing.assert_close(actual_a, a)
    torch.testing.assert_close(actual_b, b)
    with pytest.raises(ValueError, match="Incomplete"):
        convert_inkling_lora(
            {"lm_head.lora_A.weight": torch.ones(2, 5)}, {"r": 2, "lora_alpha": 2}, num_experts=3, num_shared_experts=2
        )


def test_routed_expert_conversion_preserves_each_delta():
    torch.manual_seed(11)
    count, rank, hidden, intermediate = 3, 2, 5, 4
    gate_up_a = torch.randn(count, rank, hidden, dtype=torch.float64)
    gate_up_b = torch.randn(count, 2 * intermediate, rank, dtype=torch.float64)
    down_a = torch.randn(count, rank, intermediate, dtype=torch.float64)
    down_b = torch.randn(count, hidden, rank, dtype=torch.float64)
    tensors = {
        "model.layers.0.mlp.experts.base_layer.lora_A.weight": gate_up_a.flatten(0, 1),
        "model.layers.0.mlp.experts.base_layer.lora_B.weight": gate_up_b.permute(1, 2, 0).flatten(1),
        "model.layers.0.mlp.experts.lora_A.weight": down_a.flatten(0, 1),
        "model.layers.0.mlp.experts.lora_B.weight": down_b.permute(1, 2, 0).flatten(1),
        "model.layers.1.mlp.gate_proj.lora_A.weight": torch.randn(rank, hidden),
        "model.layers.1.mlp.gate_proj.lora_B.weight": torch.randn(intermediate, rank),
        "model.layers.1.mlp.up_proj.lora_A.weight": torch.randn(rank, hidden),
        "model.layers.1.mlp.up_proj.lora_B.weight": torch.randn(intermediate, rank),
    }
    config = {
        "r": rank,
        "lora_alpha": 3 * rank,
        "target_parameters": [
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ],
    }

    output, converted_config = convert_inkling_lora(
        tensors,
        config,
        num_experts=count,
        num_shared_experts=1,
    )

    gate_b, up_b = gate_up_b.chunk(2, dim=1)
    expected = {
        "gate_proj": (gate_up_a, gate_b),
        "up_proj": (gate_up_a, up_b),
        "down_proj": (down_a, down_b),
    }
    for expert in range(count):
        for projection, (a, b) in expected.items():
            prefix = f"model.layers.0.mlp.experts.{expert}.{projection}"
            actual = output[prefix + ".lora_B.weight"] @ output[prefix + ".lora_A.weight"]
            actual *= converted_config["lora_alpha"] / converted_config["r"]
            torch.testing.assert_close(actual, 3 * (b[expert] @ a[expert]))

    assert len(output) == count * 3 * 2 + 2
    assert converted_config["r"] == 2 * rank
    assert converted_config["lora_alpha"] == 6 * rank
    assert converted_config["target_modules"] == [
        "down_proj",
        "gate_proj",
        "gate_up_proj",
        "up_proj",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA source factors")
def test_cuda_conversion_stages_packing_on_cpu():
    tensors = {
        f"model.layers.0.mlp.{proj}.lora_{factor}.weight": torch.randn(*shape, device="cuda")
        for proj in ("gate_proj", "up_proj")
        for factor, shape in (("A", (2, 5)), ("B", (3, 2)))
    }
    config = {"r": 2, "lora_alpha": 6}
    expected, expected_config = convert_inkling_lora(
        {key: tensor.cpu() for key, tensor in tensors.items()}, config, num_experts=4, num_shared_experts=2
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    actual, actual_config = convert_inkling_lora(tensors, config, num_experts=4, num_shared_experts=2)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == baseline
    assert actual_config == expected_config
    for key in expected:
        assert actual[key].device.type == "cpu"
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
