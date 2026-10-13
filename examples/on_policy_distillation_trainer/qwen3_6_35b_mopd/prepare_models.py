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
"""Download pinned models and convert only checkpoint layout, preserving teacher weights.

The published Prime-RL teachers have separate experts. Repack into the base
model's fused HF layout; only absent, unused MTP tensors come from the base.
This is an offline artifact conversion, not a training/inference code patch.
"""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

MODELS = {
    "student": ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    "caesar_cipher": (
        "semianalysisai/Qwen3.6-35B-A3B-caesar-cipher-GRPO-20260924",
        "dfedf677a5bf08a451733ec48c845c2661a6ab14",
    ),
    "simple_geometry": (
        "semianalysisai/Qwen3.6-35B-A3B-simple-geometry-GRPO-20260925",
        "679d4b66018c03249a3bfd0ecca3195ed9cd0e0d",
    ),
}
EXPERT = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def convert(source, base, destination, shard_bytes=5 * 1024**3, provenance=None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, base, destination = map(Path, (source, base, destination))
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    source_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    base_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    experts = {}
    for key in source_map:
        if match := EXPERT.fullmatch(key):
            experts.setdefault(match[1], {})[(int(match[2]), match[3])] = key

    def read(root, mapping, key):
        with safe_open(str(root / mapping[key]), framework="pt") as handle:
            return handle.get_tensor(key)

    used, weights, shards, pending = set(), {}, [], {}
    size = total = 0

    def flush():
        nonlocal pending, size
        if pending:
            name = f"model-{len(shards) + 1:05d}.safetensors"
            save_file(pending, str(stage / name), metadata={"format": "pt"})
            weights.update({key: name for key in pending})
            shards.append(name)
            pending, size = {}, 0

    for key in sorted(base_map):
        if key in source_map:
            tensor = read(source, source_map, key)
            used.add(key)
        elif key.startswith("mtp."):
            tensor = read(base, base_map, key)
        elif key.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
            prefix, projection = key.rsplit(".", 1)
            group = experts[prefix]
            count = max(index for index, _ in group) + 1
            names = ("gate_proj", "up_proj") if projection == "gate_up_proj" else ("down_proj",)
            stacked = [torch.stack([read(source, source_map, group[i, part]) for i in range(count)]) for part in names]
            tensor = torch.cat(stacked, dim=1) if len(stacked) == 2 else stacked[0]
            used.update(group[i, part] for i in range(count) for part in names)
        else:
            raise ValueError(f"Missing trained tensor: {key}")
        with safe_open(str(base / base_map[key]), framework="pt") as handle:
            if list(tensor.shape) != handle.get_slice(key).get_shape():
                raise ValueError(f"Shape mismatch: {key}")
        pending[key] = tensor.contiguous()
        size += tensor.numel() * tensor.element_size()
        total += tensor.numel() * tensor.element_size()
        if size >= shard_bytes:
            flush()
    flush()
    if set(source_map) != used:
        raise ValueError(f"Unused teacher tensors: {sorted(set(source_map) - used)[:10]}")
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json":
            shutil.copyfile(path, stage / path.name)
    (stage / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total},
                "weight_map": weights,
            },
            indent=2,
        )
    )
    (stage / "conversion.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "base": str(base),
                "tensors": len(weights),
                "conversion": "Separate experts packed into gate_up/down tensors; base MTP only; runtime MTP disabled",
                "provenance": provenance,
                "index_sha256": hashlib.sha256((stage / "model.safetensors.index.json").read_bytes()).hexdigest(),
                "shard_sizes": {name: (stage / name).stat().st_size for name in shards},
            },
            indent=2,
        )
    )
    stage.rename(destination)


def verify_conversion(destination, provenance):
    """Only reuse atomically completed conversions of these exact checkpoint revisions."""
    destination = Path(destination)
    receipt = json.loads((destination / "conversion.json").read_text())
    if receipt.get("provenance") != provenance:
        raise ValueError(f"Converted checkpoint provenance mismatch: {destination}")
    index_path = destination / "model.safetensors.index.json"
    if receipt.get("index_sha256") != hashlib.sha256(index_path.read_bytes()).hexdigest():
        raise ValueError(f"Converted checkpoint index changed: {destination}")
    shards = set(json.loads(index_path.read_text())["weight_map"].values())
    if shards != set(receipt["shard_sizes"]):
        raise ValueError(f"Converted checkpoint shard inventory mismatch: {destination}")
    for name, size in receipt["shard_sizes"].items():
        if (destination / name).stat().st_size != size:
            raise ValueError(f"Converted checkpoint shard truncated: {destination / name}")


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("assets", type=Path)
    args = parser.parse_args()
    for role, (repo, revision) in MODELS.items():
        path = args.assets / ("student" if role == "student" else f"raw/{role}")
        snapshot_download(repo_id=repo, revision=revision, local_dir=path)
        if role != "student":
            destination = args.assets / "teachers" / role
            provenance = {"teacher": list(MODELS[role]), "base": list(MODELS["student"])}
            if destination.exists():
                verify_conversion(destination, provenance)
            else:
                convert(path, args.assets / "student", destination, provenance=provenance)
    manifest = args.assets / "models.json.partial"
    manifest.write_text(json.dumps(MODELS, indent=2) + "\n")
    manifest.replace(args.assets / "models.json")


if __name__ == "__main__":
    main()
