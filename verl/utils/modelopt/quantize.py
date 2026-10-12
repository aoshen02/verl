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

"""ModelOpt NVFP4 quantization config and application for Megatron QAT."""

import copy
import json
import os
import re
import weakref

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg
from modelopt.torch.quantization.nn import TensorQuantizer

_NVFP4_W4A16_QUANTIZER_CFG = {
    "*weight_quantizer": {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    },
    "*input_quantizer": {"enable": False},
}


def _ignore_patterns_to_quant_cfg(ignore_patterns: list[str]) -> list[dict]:
    cfg = []
    mapping = {
        "lm_head": "*output_layer*",
        "*mlp.gate": "*router*",
        "*self_attn*": "*self_attention*",
    }
    for pattern in ignore_patterns:
        key = pattern
        if key in mapping:
            key = mapping[key]
        cfg.append({"quantizer_name": key, "enable": False})
    return cfg


def build_quantize_config(
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> dict:
    """Build a complete ModelOpt quantization config for ``mtq.quantize``."""
    if qat_mode != "w4a16":
        raise ValueError(f"Only 'w4a16' is supported, got: {qat_mode}")

    if ignore_patterns is None:
        ignore_patterns = []

    ignore_cfg = _ignore_patterns_to_quant_cfg(ignore_patterns)

    quant_cfg = mtq.normalize_quant_cfg_list(_NVFP4_W4A16_QUANTIZER_CFG)
    disabled_cfg = copy.deepcopy(_default_disabled_quantizer_cfg)
    if isinstance(disabled_cfg, dict):
        disabled_cfg = mtq.normalize_quant_cfg_list(disabled_cfg)
    quant_cfg.extend(disabled_cfg)
    quant_cfg.extend(ignore_cfg)
    return {"quant_cfg": quant_cfg, "algorithm": "max"}


# ModelOpt's recipe for the Nemotron-H Lightning NVFP4 deployment: static NVFP4 4/6 (MSE) routed and
# shared experts and lm_head, FP8 Mamba in/out_proj weights and inputs, FP8 KV cache, MTP disabled.
_LIGHTNING_RECIPE = "models/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/ptq/w4a16_nvfp4_4o6"

# Filled by apply_mixed_precision_qat for this rank; read by mixed_precision_export.
#   scales: HF module -> {"algo", "expert", "scales": (...)} of the weights quantized on this rank
#   static: HF module -> {suffix: tensor} static activation scales (input_scale / k_scale / v_scale)
#   local:  HF module -> (weight quantizer, weight parameter) for the forward-vs-deployed check
#   act:    quantizer name -> static FP8 input / KV quantizer
DEPLOY_STATE: dict = {"scales": {}, "static": {}, "local": {}, "act": {}}


# Deployment scales per weight quantizer. Not an attribute of the quantizer: ModelOpt pickles every
# quantizer attribute into the module's ``_extra_state``, once per expert for grouped linears.
_DEPLOY: "weakref.WeakKeyDictionary[nn.Module, tuple]" = weakref.WeakKeyDictionary()


def state_digest() -> dict[str, str]:
    """Digests of this rank's quantizer state: live amax buffers, the scales the forward uses, activation amax."""
    import hashlib

    def digest(tensors):
        h = hashlib.sha256()
        for t in tensors:
            t = t.detach().contiguous().cpu()
            h.update(str((t.dtype, tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()[:16]

    amax, scales = [], []
    for name in sorted(DEPLOY_STATE["local"]):
        quantizer = DEPLOY_STATE["local"][name][0]
        amax.append(quantizer._amax)
        if getattr(quantizer, "_global_amax", None) is not None:
            amax.append(quantizer._global_amax)
        scales.extend(_DEPLOY[quantizer][1])
    act = [DEPLOY_STATE["act"][name]._amax for name in sorted(DEPLOY_STATE["act"])]
    return {"weight_amax": digest(amax), "deploy_scales": digest(scales), "act_amax": digest(act)}


def _dequantize(algorithm: str, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """FP32 value of deployment tensors (ModelOpt's reference NVFP4 dequantize, not its Triton fast path)."""
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    if algorithm == "FP8":
        return tensors["weight"].float() * tensors["weight_scale"].float()
    packed = tensors["weight"]
    shape = torch.Size((*packed.shape[:-1], packed.shape[-1] * 2))
    return NVFP4QTensor(shape, torch.float32, packed).dequantize(
        dtype=torch.float32,
        scale=tensors["weight_scale"],
        double_scale=tensors["weight_scale_2"].float().reshape(()),
        block_sizes={-1: 16},
    )


def official_scales(algorithm: str, quantizer, weight: torch.Tensor) -> tuple:
    """The deployment scales ModelOpt's exporter derives from a calibrated (static) weight quantizer."""
    from modelopt.torch.export.quant_utils import get_scaling_factor
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    if algorithm == "FP8":
        return (get_scaling_factor(quantizer).reshape(()),)
    scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(quantizer).reshape(())
    scale = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(quantizer, weight, scale_2)[0]
    return scale, scale_2


def official_quantize(algorithm: str, weight: torch.Tensor, scales: tuple) -> dict[str, torch.Tensor]:
    """ModelOpt's exporter real quantization (``to_quantized_weight``) with the static scales."""
    from modelopt.torch.export.quant_utils import QUANTIZATION_W4A16_NVFP4, to_quantized_weight

    weight = weight.to(torch.bfloat16)
    if algorithm == "FP8":
        (scale,) = scales
        # to_quantized_weight's FP8 rule, E4M3(BF16(w / scale)), written out: on the GPU PyTorch rounds
        # its 0-dim scale to BF16 before dividing, which the exported checkpoint (CPU) does not.
        quotient = (weight.float() / scale.float()).to(torch.bfloat16)
        code = quotient.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return {"weight": code, "weight_scale": scale}
    scale, scale_2 = scales
    packed = to_quantized_weight(weight, scale, QUANTIZATION_W4A16_NVFP4, scale_2, 16)
    return {"weight": packed, "weight_scale": scale, "weight_scale_2": scale_2}


def _deployed_ste_forward(quantizer, inputs):
    """Forward value = the deployed weight (dequantized exporter output); gradient straight through.

    ModelOpt's fake-quant kernels are not bit-identical to its exporter (FP8: FP32 vs BF16 quotient;
    NVFP4: rounding at ties), so the trained weight would differ from the served one.
    """
    with torch.no_grad():
        algorithm, scales = _DEPLOY[quantizer]
        deployed = _dequantize(algorithm, official_quantize(algorithm, inputs.detach(), scales))
    return inputs + (deployed.to(inputs.dtype) - inputs).detach()


def _deployment_modules(core: nn.Module, ep_rank: int):
    """Yield (kind, HF module name, mcore module name, module[, local expert]) of the deployment layers."""
    for name, module in core.named_modules():
        if name.startswith("mtp.") or ".mtp." in name:
            continue
        match = re.search(r"decoder\.layers\.(\d+)\.(.+)$", name)
        if name == "output_layer":
            yield "nvfp4", "lm_head", name, module, None
        if match is None:
            continue
        layer, rest = match.groups()
        hf = f"backbone.layers.{layer}.mixer"
        if rest in ("mixer.in_proj", "mixer.out_proj"):
            yield "fp8", f"{hf}.{rest.split('.')[1]}", name, module, None
        elif rest in ("mlp.shared_experts.linear_fc1", "mlp.shared_experts.linear_fc2"):
            yield (
                "nvfp4",
                f"{hf}.shared_experts.{'up_proj' if rest.endswith('fc1') else 'down_proj'}",
                name,
                module,
                None,
            )
        elif rest in ("mlp.experts.linear_fc1", "mlp.experts.linear_fc2"):
            proj = "up_proj" if rest.endswith("fc1") else "down_proj"
            for local in range(module.num_gemms):
                yield "nvfp4", f"{hf}.experts.{ep_rank * module.num_gemms + local}.{proj}", name, module, local
        elif rest == "self_attention.core_attention":
            yield "kv", hf, name, module, None


def apply_mixed_precision_qat(model: nn.Module, deploy_path: str) -> nn.Module:
    """ModelOpt's Lightning recipe on the MCore model, forward on the deployed weights.

    Weights: static MSE calibration of the recipe. FP8 input / KV quantizers: the deployment
    checkpoint's static scales, pinned through the recipe's ``constant_amax`` (no calibration data).
    """
    import modelopt.torch.quantization as mtq
    from megatron.core import parallel_state as mpu
    from megatron.core.utils import unwrap_model
    from modelopt.recipe import load_recipe
    from modelopt.torch.export.quant_utils import get_kv_cache_scaling_factor, get_scaling_factor
    from safetensors import safe_open

    core = unwrap_model(model)
    ep_rank = mpu.get_expert_model_parallel_rank()
    with open(os.path.join(deploy_path, "hf_quant_config.json")) as f:
        deployed_layers = json.load(f)["quantization"]["quantized_layers"]
    with open(os.path.join(deploy_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    def checkpoint_scalar(key):
        with safe_open(os.path.join(deploy_path, weight_map[key]), framework="pt") as f:
            return float(f.get_tensor(key).double().reshape(()))

    # (quantizer name, checkpoint key) of the static activation scales
    static_keys = []
    for kind, hf, name, _module, _local in _deployment_modules(core, ep_rank):
        if kind == "fp8":
            static_keys.append((f"{name}.input_quantizer", f"{hf}.input_scale"))
        elif kind == "kv":
            static_keys.append((f"{name}.k_bmm_quantizer", f"{hf}.k_proj.k_scale"))
            static_keys.append((f"{name}.v_bmm_quantizer", f"{hf}.v_proj.v_scale"))
    config = load_recipe(_LIGHTNING_RECIPE).quantize.model_dump()
    config["quant_cfg"] = list(config["quant_cfg"]) + [
        {
            "quantizer_name": quantizer,
            "enable": True,
            "cfg": {"num_bits": (4, 3), "axis": None, "constant_amax": checkpoint_scalar(key) * 448.0},
        }
        for quantizer, key in static_keys
    ]
    # Weight calibration is rank-local and the activation amax is constant: nothing to sync.
    config["algorithm"] = {**config["algorithm"], "distributed_sync": False}
    mtq.quantize(core, config, forward_loop=None)
    # The quantizers are rebuilt from the recipe at every start, so a checkpoint load only has to fill
    # their buffers. ModelOpt's restore callback (meant for an unquantized model, run before the weight
    # load) would rebuild them inside the same load and drop the Mamba FP8 amax.
    for module in core.modules():
        if hasattr(module, "allow_post_restore"):
            module.allow_post_restore = False

    state = {"scales": {}, "static": {}, "local": {}, "act": {}}
    modules = dict(core.named_modules())
    for quantizer, _key in static_keys:
        state["act"][quantizer] = modules[quantizer]
    mtp = [
        (n, m) for n, m in modules.items() if isinstance(m, TensorQuantizer) and (n.startswith("mtp.") or ".mtp." in n)
    ]
    if any(m.is_enabled for _, m in mtp):
        raise RuntimeError(f"MTP quantizers enabled: {[n for n, m in mtp if m.is_enabled][:5]}")
    quantized = set()
    for kind, hf, name, module, local in _deployment_modules(core, ep_rank):
        if kind == "kv":
            k_scale, v_scale = get_kv_cache_scaling_factor(module, clamp_fp8_scales=False)
            state["static"][f"{hf}.k_proj"] = {"k_scale": k_scale.float().cpu()}
            state["static"][f"{hf}.v_proj"] = {"v_scale": v_scale.float().cpu()}
            continue
        algorithm = "FP8" if kind == "fp8" else "W4A16_NVFP4"
        if deployed_layers.get(hf, {}).get("quant_algo") != algorithm:
            raise RuntimeError(f"{name}: deployment lists {deployed_layers.get(hf)} for {hf}, recipe gives {algorithm}")
        quantizer = module.weight_quantizer if local is None else module.weight_quantizer[local]
        weight = module.weight if local is None else getattr(module, f"weight{local}")
        if not quantizer.is_enabled or getattr(quantizer, "_amax", None) is None:
            raise RuntimeError(f"{name}: the recipe left {hf} without a calibrated weight quantizer")
        with torch.no_grad():
            scales = official_scales(algorithm, quantizer, weight)
        _DEPLOY[quantizer] = (algorithm, scales)
        quantizer.forward = _deployed_ste_forward.__get__(quantizer)
        state["scales"][hf] = {"algo": algorithm, "expert": local is not None, "scales": tuple(s.cpu() for s in scales)}
        state["local"][hf] = (quantizer, weight)
        quantized.add(hf)
        if kind == "fp8":
            state["static"][hf] = {"input_scale": get_scaling_factor(module.input_quantizer).float().cpu()}
    ep_size = mpu.get_expert_model_parallel_world_size()
    expected = {
        hf
        for hf in deployed_layers
        if ".experts." not in hf
        or re.search(r"\.experts\.(\d+)\.", hf) is None
        or int(re.search(r"\.experts\.(\d+)\.", hf).group(1)) * ep_size // _num_experts(deployed_layers) == ep_rank
    }
    if quantized != expected:
        raise RuntimeError(
            f"recipe/deployment mismatch on EP rank {ep_rank}: missing {sorted(expected - quantized)[:5]}, "
            f"extra {sorted(quantized - expected)[:5]}"
        )
    for quantizer, key in static_keys:
        module_key, _, suffix = key.rpartition(".")
        got = state["static"][module_key][suffix].reshape(()).item()
        want = torch.tensor(checkpoint_scalar(key), dtype=torch.float32).item()
        if got != want:
            raise RuntimeError(f"{quantizer}: static scale {got} != deployment {key} {want}")
    DEPLOY_STATE.update(state)
    print(
        f"[QAT modelopt_mixed] EP rank {ep_rank}: {len(quantized)} weight quantizers "
        f"(static, forward = deployed weight), {len(static_keys)} static activation/KV quantizers, "
        f"MTP quantizers enabled 0 of {len(mtp)}",
        flush=True,
    )
    if os.environ.get("VERL_QAT_VERIFY_DEPLOYED") == "1":
        print(f"[QAT-STATE rank={torch.distributed.get_rank()} init] {state_digest()}", flush=True)
    return model


def _num_experts(deployed_layers: dict) -> int:
    return 1 + max(int(m.group(1)) for m in (re.search(r"\.experts\.(\d+)\.", k) for k in deployed_layers) if m)


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
    deploy_path: str | None = None,
) -> nn.Module:
    """Apply Quantization-Aware Training to a Megatron model."""
    if qat_mode == "modelopt_mixed":
        return apply_mixed_precision_qat(model, deploy_path)
    config = build_quantize_config(qat_mode, ignore_patterns)
    mtq.quantize(model, config)
    return model
