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

"""CPU tests for the receiver-side weight sync in ``vLLMColocateWorkerExtension``.

With LoRA, the base sync arrives in PEFT naming (``*.base_layer.*``) and vLLM loads
plain names, so the receiver strips one ``.base_layer.``; LoRA adapters go to vLLM's
``LoRARequest`` unchanged. Deps are stubbed, so no GPU or vLLM is needed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_weight_update_utils():
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeLoRARequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_vllm_rollout_utils(weight_update_utils):
    """Load ``vllm_rollout/utils.py`` with heavyweight deps stubbed (no GPU/vLLM needed)."""
    module_name = "verl.workers.rollout.vllm_rollout.utils"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/utils.py"

    fake_outputs = types.ModuleType("vllm.outputs")
    fake_outputs.RequestOutput = type("RequestOutput", (), {})
    fake_lora_request = types.ModuleType("vllm.lora.request")
    fake_lora_request.LoRARequest = _FakeLoRARequest
    fake_lora = types.ModuleType("vllm.lora")
    fake_lora.request = fake_lora_request
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.outputs = fake_outputs
    fake_vllm.lora = fake_lora

    fake_vllm_third_party = types.ModuleType("verl.third_party.vllm")
    fake_vllm_third_party.VLLM_SLEEP_LEVEL = 1
    fake_vllm_third_party.get_version = lambda pkg: "0.8.0"

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")
    fake_vllm_utils.TensorLoRARequest = type("TensorLoRARequest", (), {})
    fake_vllm_utils.VLLMHijack = type("VLLMHijack", (), {"hijack": staticmethod(lambda: None)})

    fake_bucketed_transfer = types.ModuleType("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
    fake_bucketed_transfer.BucketedWeightReceiver = None

    # NOTE: deliberately do NOT stub verl.plugin.platform (see test_vllm_weight_update_utils_on_cpu.py).
    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "vllm.lora": fake_lora,
        "vllm.lora.request": fake_lora_request,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer": fake_bucketed_transfer,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": weight_update_utils,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_vllm_rollout_utils = _load_vllm_rollout_utils(_load_weight_update_utils())
vLLMColocateWorkerExtension = _vllm_rollout_utils.vLLMColocateWorkerExtension


class _FakeModel:
    """A fake vLLM model exposing only the surface the receiver uses."""

    def __init__(self):
        self.loaded = []

    def named_buffers(self):
        return iter(())

    def load_weights(self, weights):
        self.loaded.extend(name for name, _ in weights)


def _make_worker(model):
    worker = object.__new__(vLLMColocateWorkerExtension)
    vllm_config = types.SimpleNamespace(speculative_config=None, model_config=None)
    worker.model_runner = types.SimpleNamespace(model=model, vllm_config=vllm_config, drafter=None)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-bucketed-weight-sync.sock"
    return worker


@pytest.mark.parametrize(
    "peft_config,incoming,expected",
    [
        (None, "fc.base_layer.weight", "fc.base_layer.weight"),
        ({"r": 1}, "merger.linear_fc1.base_layer.weight", "merger.linear_fc1.weight"),
        ({"r": 1}, "q_proj.base_layer.bias", "q_proj.bias"),
        ({"r": 1}, "embed_tokens.weight", "embed_tokens.weight"),
        ({"r": 1}, "a.base_layer.base_layer.weight", "a.base_layer.weight"),
    ],
    ids=["no-peft-unchanged", "peft-strip-weight", "peft-strip-bias", "peft-plain-name", "peft-strip-once"],
)
def test_base_sync_strips_peft_base_layer(peft_config, incoming, expected):
    model = _FakeModel()
    _make_worker(model)._update_weights([(incoming, torch.empty(0))], peft_config=peft_config, base_sync_done=False)
    assert model.loaded == [expected]


def test_lora_adapter_goes_to_vllm_lora_request_unchanged():
    added = []
    worker = _make_worker(_FakeModel())
    worker.add_lora = added.append
    name = "language_model.model.layers.0.self_attn.q_proj.lora_A.weight"

    worker._update_weights([(name, torch.ones(1))], peft_config={"r": 1}, base_sync_done=True)

    # Off NPU the adapter tensors go straight to vLLM's LoRARequest(lora_tensors=..., peft_config=...)
    assert len(added) == 1 and isinstance(added[0], _FakeLoRARequest)
    assert set(added[0].lora_tensors) == {name} and added[0].peft_config == {"r": 1}


class _FakeBucketReceiver:
    """Drives the on_bucket_received callback with a scripted bucket sequence."""

    def __init__(self, buckets):
        self._buckets = buckets

    def receive_weights(self, on_bucket_received):
        for weights, is_last in self._buckets:
            on_bucket_received(weights, is_last)


def _install_fake_receiver(monkeypatch, buckets):
    monkeypatch.setattr(_vllm_rollout_utils, "BucketedWeightReceiver", lambda *a, **k: _FakeBucketReceiver(buckets))


def _install_fake_reload(monkeypatch, events):
    """Record vLLM's reload lifecycle calls (start_reload / finish_reload)."""
    reload = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload.start_reload = lambda model: events.append("start_reload")
    reload.finish_reload = lambda model, model_config, device: events.append("finish_reload")
    monkeypatch.setitem(sys.modules, reload.__name__, reload)


def test_update_weights_from_ipc_accumulates_lora_across_buckets(monkeypatch):
    """A LoRA adapter split across two buckets yields one add_lora with all tensors."""
    _install_fake_receiver(
        monkeypatch, [([("lora.A.weight", torch.ones(1))], False), ([("lora.B.weight", torch.zeros(1))], True)]
    )
    added = []
    worker = _make_worker(_FakeModel())
    worker.add_lora = lambda request: added.append(dict(request.lora_tensors))
    worker.remove_lora = lambda lora_id: None

    worker.update_weights_from_ipc(peft_config={"r": 1}, base_sync_done=True)

    # One add_lora with both buckets' tensors; the first is cloned, not a view into a reused buffer.
    assert len(added) == 1
    assert set(added[0]) == {"lora.A.weight", "lora.B.weight"}
    torch.testing.assert_close(added[0]["lora.A.weight"], torch.ones(1))


def test_standard_sync_loads_between_start_and_finish_reload(monkeypatch):
    """The tensors handed to load_weights are the receiver's views into its reused buffer."""
    events = []
    _install_fake_reload(monkeypatch, events)
    bucket_buffer = torch.zeros(2)
    views = [bucket_buffer[0:1], bucket_buffer[1:2]]
    _install_fake_receiver(monkeypatch, [([("q.weight", views[0])], False), ([("k.weight", views[1])], True)])
    model = _FakeModel()
    loaded = []

    def _load(weights):
        for name, tensor in weights:
            events.append(f"load:{name}")
            loaded.append(tensor)

    model.load_weights = _load
    _make_worker(model).update_weights_from_ipc(peft_config=None, base_sync_done=False)

    assert events == ["start_reload", "load:q.weight", "load:k.weight", "finish_reload"]
    assert [t.data_ptr() for t in loaded] == [v.data_ptr() for v in views]


def test_lora_adapter_sync_skips_the_reload(monkeypatch):
    events = []
    _install_fake_reload(monkeypatch, events)
    _install_fake_receiver(monkeypatch, [([("lora.A.weight", torch.ones(1))], True)])
    worker = _make_worker(_FakeModel())
    worker.add_lora = lambda request: events.append("add_lora")
    worker.remove_lora = lambda lora_id: None

    worker.update_weights_from_ipc(peft_config={"r": 1}, base_sync_done=True)

    assert events == ["add_lora"]
