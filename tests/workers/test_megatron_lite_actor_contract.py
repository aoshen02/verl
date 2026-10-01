# Copyright 2026 aoshen02
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

"""Actor indexing and normalization contracts, not a model/RL quality test."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("megatron.lite.runtime.contracts")

from verl.utils import tensordict_utils as tu
from verl.workers.engine.megatron_lite import engine as implementation
from verl.workers.engine.megatron_lite.data import pack_actor_batch, unpack_logprobs
from verl.workers.engine.megatron_lite.engine import MegatronLiteEngine
from verl.workers.utils.padding import no_padding_2_padding


@pytest.fixture
def actor_data():
    def nested(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    prompts = [torch.tensor([10, 11, 12]), torch.tensor([20, 21, 22, 23])]
    responses = [torch.tensor([13, 14]), torch.tensor([24, 25, 26])]
    rows = [torch.cat(pair) for pair in zip(prompts, responses, strict=True)]
    return TensorDict(
        {
            "input_ids": nested(rows),
            "prompts": nested(prompts),
            "responses": nested(responses),
            "response_mask": nested([torch.tensor([1, 1]), torch.tensor([1, 1, 0])]),
            "position_ids": nested([torch.arange(len(row)) for row in rows]),
        },
        batch_size=[2],
    )


def test_packing_preserves_response_shift_and_gradient_positions(actor_data):
    batch = pack_actor_batch(actor_data)
    assert batch.seq_lens.tolist() == [5, 7]
    assert batch.loss_mask.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    scores = torch.arange(12, dtype=torch.float32, requires_grad=True)
    response = no_padding_2_padding(unpack_logprobs(scores, batch), actor_data)
    assert torch.equal(response, torch.tensor([[2.0, 3.0, 0.0], [8.0, 9.0, 10.0]]))
    response.sum().backward()
    assert scores.grad.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]


def test_packing_rejects_different_sample_tokens(actor_data):
    rows = list(actor_data["input_ids"].unbind())
    actor_data["input_ids"] = torch.nested.as_nested_tensor([rows[0], rows[1].flip(0)], layout=torch.jagged)
    with pytest.raises(ValueError, match="prompt/response"):
        pack_actor_batch(actor_data)


class ScalarRuntime:
    def __init__(self):
        self.weight = torch.tensor(0.5, requires_grad=True)
        self.zero_calls = self.step_calls = 0

    def zero_grad(self, handle):
        self.zero_calls += 1
        self.weight.grad = None

    def optimizer_step(self, handle):
        self.step_calls += 1
        return True, float(self.weight.grad.abs()), 0

    def forward_backward(self, handle, batches, callback, *, num_microbatches, forward_only):
        for batch, context in batches:
            scores = self.weight * batch.input_ids.float()
            loss, _ = callback({"log_probs": scores[None], "loss": scores.sum()}, batch, context)
            if not forward_only:
                (loss / num_microbatches).backward()


@pytest.mark.parametrize("micro_size", [1, 2])
def test_train_batch_normalizes_once_and_steps_once(actor_data, micro_size, monkeypatch):
    engine = object.__new__(MegatronLiteEngine)
    engine.runtime = ScalarRuntime()
    engine.handle = SimpleNamespace(dp_group=None)
    engine.dp_size = 1
    engine.output_rank = True
    engine.vllm_config = None
    monkeypatch.setattr(implementation.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(implementation, "set_current_vllm_config", lambda config: nullcontext())
    tu.assign_non_tensor(actor_data, use_dynamic_bsz=False, micro_batch_size_per_gpu=micro_size)

    def loss_function(model_output, data, dp_group):
        scores = no_padding_2_padding(model_output["log_probs"], data)
        mask = data["response_mask"].to_padded_tensor(0)
        return (scores * mask).sum() / data["batch_num_tokens"], {}

    engine.train_batch(actor_data, loss_function)
    assert engine.runtime.weight.grad.item() == 18.0
    assert engine.runtime.zero_calls == engine.runtime.step_calls == 1
