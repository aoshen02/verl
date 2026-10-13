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
"""Fail before allocating Ray pools if assets or runtime disagree with the recipe."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from .evaluate_teachers import prepare_prompts
    from .prepare_models import MODELS

    root = Path(os.environ["VERL_ROOT"])
    expected_lock = Path("/opt/verl-uv-lock.sha256").read_text().split()[0]
    if hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest() != expected_lock:
        raise ValueError("Mounted checkout's uv.lock differs from the container build")
    for package, expected in {"vllm": "0.29.0", "transformers": "5.12.1", "reasoning-gym": "0.1.25"}.items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Wrong {package} version; rebuild the image")
    if torch.__version__.split("+")[0] != "2.13.0":
        raise ValueError("Expected the pinned torch 2.13.0 runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise ValueError("Expected eight visible GPUs on each Slurm node")
    devices = [torch.cuda.get_device_name(i) for i in range(8)]
    if any("B200" not in name for name in devices):
        raise ValueError("This example's capacity and FLOP denominator are for B200")
    assets = Path(os.environ["MOPD_ASSETS"])
    if json.loads((assets / "models.json").read_text()) != {k: list(v) for k, v in MODELS.items()}:
        raise ValueError("Model asset provenance differs from the pinned checkpoints")
    tokenizer = AutoTokenizer.from_pretrained(os.environ["STUDENT_MODEL"])
    for name in ("STUDENT_MODEL", "CAESAR_TEACHER_MODEL", "GEOMETRY_TEACHER_MODEL"):
        path = Path(os.environ[name])
        if AutoConfig.from_pretrained(path).model_type != "qwen3_5_moe":
            raise ValueError(f"Unexpected model architecture: {path}")
        model_tokenizer = AutoTokenizer.from_pretrained(path)
        if model_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("Teacher/student token IDs must have identical meanings")
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        if any(".experts.0." in key for key in index):
            raise ValueError("Teacher experts have not been converted to the base HF layout")
        for shard in set(index.values()):
            if not (path / shard).is_file():
                raise FileNotFoundError(path / shard)
    data = Path(os.environ["MOPD_DATA"])
    manifest = json.loads((data / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = data / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["parquet_sha256"]:
            raise ValueError(f"Dataset changed: {name}")
        rows = pq.read_table(path).to_pylist()
        prompts = prepare_prompts(tokenizer, rows)
        print(f"Validated {name}: {len(prompts)} prompts, longest={max(map(len, prompts))} tokens", flush=True)
    # Import validates the pinned Ray worker extension before any model is loaded.
    from hydra import compose, initialize_config_module

    from .metrics.runtime import (
        MeasuredWorker,  # noqa: F401
        validate_teacher_loading,
    )

    with initialize_config_module(config_module=__package__, version_base=None):
        validate_teacher_loading(compose(config_name="config"))

    inventory = {
        "devices": devices,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    Path(os.environ["MOPD_OUTPUT"], "runtime.json").write_text(json.dumps(inventory, indent=2))
    print("Preflight passed; GPU model loading and the long-context update still require a smoke run.")


if __name__ == "__main__":
    main()
