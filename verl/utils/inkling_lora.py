# Copyright 2026 individual contributors.
# SPDX-License-Identifier: Apache-2.0
"""Lossless PEFT factor packing for Inkling's serving projections."""

import copy

import torch
import torch.nn.functional as F


def unpack_experts(a, b, count):
    """Decode PEFT ParamWrapper's flattened expert axes without forming delta-W."""
    if a.shape[0] % count or b.shape[1] != a.shape[0]:
        raise ValueError("Invalid PEFT expert factor shapes")
    a = a.reshape(count, -1, a.shape[-1])
    b = b.reshape(b.shape[0], -1, count).permute(2, 0, 1)
    return a, b


def pack_shared(a, b, *, down):
    """Pack independent experts into one linear adapter, preserving every factor."""
    if down:
        return torch.block_diag(*a.unbind()), torch.cat(b.unbind(), dim=1)
    return torch.cat(a.unbind(), dim=0), torch.block_diag(*b.unbind())


def pack_dense(gate, up):
    """Combine independent gate/up adapters in interleaved checkpoint row order."""
    ag, bg = gate
    au, bu = up
    if bg.shape[0] != bu.shape[0]:
        raise ValueError("Gate/up output widths differ")
    b = torch.block_diag(bg, bu)
    rows = torch.arange(b.shape[0], device=b.device).reshape(2, -1).T.flatten()
    return torch.cat((ag, au), dim=0), b[rows]


def convert_inkling_lora(tensors, config, *, num_experts, num_shared_experts):
    """Return serving factors/config; requires vLLM's stacked MoE LoRA format.

    Packing grows the serving rank, not the trainable rank. Alpha grows by the
    same ratio, preserving alpha/r. No base weights or full delta-W are gathered.
    Stage factors on CPU before packing, matching vLLM's adapter loader device.
    """
    if config.get("use_rslora") or config.get("use_dora") or config.get("rank_pattern"):
        raise ValueError("Only uniform-rank standard LoRA is supported")
    if config.get("alpha_pattern") or config.get("bias", "none") != "none":
        raise ValueError("Only uniform-alpha, bias-free LoRA is supported")
    for group, projections in (
        ("experts", ["gate_up_proj", "down_proj"]),
        ("shared_experts", ["gate_proj", "up_proj", "down_proj"]),
    ):
        if any(f".mlp.{group}." in key for key in tensors):
            targets = [key for key in config.get("target_parameters", []) if f"mlp.{group}." in key]
            if targets != [f"mlp.{group}.{projection}" for projection in projections]:
                raise ValueError(f"Unsupported PEFT wrapper order for {group}: {targets}")
    pairs = {}
    for key, value in tensors.items():
        if key.endswith("lm_head.base_layer.weight"):
            continue
        name, marker, suffix = key.rpartition(".lora_")
        if not marker or suffix not in ("A.weight", "B.weight"):
            raise ValueError(f"Unexpected adapter key: {key}")
        name = name.removeprefix("base_model.model.")
        name = name.replace("model.language_model.layers.", "model.layers.")
        pairs.setdefault(name, {})[suffix[0]] = value.cpu()
    if any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Incomplete adapter A/B pair")
    pairs = {name: (pair["A"], pair["B"]) for name, pair in pairs.items()}
    result = {}

    def emit(name, pair):
        if name in result:
            raise ValueError(f"Duplicate serving module: {name}")
        result[name] = pair

    attention = dict(q_proj="wq_du", k_proj="wk_dv", v_proj="wv_dv", r_proj="wr_du", o_proj="wo_ud")
    while pairs:
        name = next(iter(pairs))
        pair = pairs.pop(name)
        if ".self_attn." in name:
            parent, projection = name.rsplit(".self_attn.", 1)
            emit(f"{parent}.attn.{attention[projection]}", pair)
        elif name.endswith(".mlp.gate_proj") or name.endswith(".mlp.up_proj"):
            parent = name.rsplit(".", 1)[0]
            gate = pair if name.endswith(".gate_proj") else pairs.pop(parent + ".gate_proj")
            up = pair if name.endswith(".up_proj") else pairs.pop(parent + ".up_proj")
            emit(parent + ".gate_up_proj", pack_dense(gate, up))
        elif ".mlp.shared_experts" in name:
            parent, suffix = name.split(".shared_experts", 1)
            projection = {".base_layer.base_layer": "w1", ".base_layer": "w3", "": "w2"}[suffix]
            a, b = unpack_experts(*pair, num_shared_experts)
            emit(f"{parent}.sink_experts.{projection}", pack_shared(a, b, down=projection == "w2"))
        elif ".mlp.experts" in name:
            parent, suffix = name.split(".experts", 1)
            a, b = unpack_experts(*pair, num_experts)
            if suffix == ".base_layer":
                gate, up = b.chunk(2, dim=1)
                for expert in range(num_experts):
                    prefix = f"{parent}.experts.{expert}"
                    emit(prefix + ".gate_proj", (a[expert], gate[expert]))
                    emit(prefix + ".up_proj", (a[expert], up[expert]))
            elif suffix == "":
                for expert in range(num_experts):
                    emit(
                        f"{parent}.experts.{expert}.down_proj",
                        (a[expert], b[expert]),
                    )
            else:
                raise ValueError(f"Unknown expert wrapper: {name}")
        elif name == "lm_head" or name.endswith(".mlp.down_proj"):
            emit(name, pair)
        else:
            raise ValueError(f"Unsupported adapter module: {name}")
    rank = max(a.shape[-2] for a, _ in result.values())
    updated = copy.deepcopy(config)
    updated.update(
        r=rank,
        lora_alpha=config["lora_alpha"] * rank / config["r"],
        target_modules=sorted({name.rsplit(".", 1)[-1] for name in result}),
        target_parameters=None,
    )
    packed = {}
    for name, (a, b) in result.items():
        packed[f"{name}.lora_A.weight"] = F.pad(a, (0, 0, 0, rank - a.shape[-2])).contiguous()
        packed[f"{name}.lora_B.weight"] = F.pad(b, (0, rank - b.shape[-1])).contiguous()
    return packed, updated
