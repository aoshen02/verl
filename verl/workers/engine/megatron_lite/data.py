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

"""Preserve verl sample boundaries when packing Megatron-lite actor inputs."""

import torch
from megatron.lite.runtime.contracts import PackedBatch
from tensordict import TensorDict


def pack_actor_batch(data: TensorDict) -> PackedBatch:
    """Pack unpadded actor samples without altering label or position semantics."""
    ids = data["input_ids"]
    if not ids.is_nested or ids.layout != torch.jagged or ids.dim() != 2:
        raise ValueError("Megatron-lite requires jagged unpadded input_ids")
    rows = list(ids.unbind())
    if not rows or any(row.numel() < 2 for row in rows):
        raise ValueError("Each actor sequence needs at least two tokens")
    prompts, responses = data["prompts"], data["responses"]
    mask = data["response_mask"]
    if not prompts.is_nested or not responses.is_nested or not mask.is_nested:
        raise ValueError("Prompts, responses and response_mask must be jagged")
    prompt_rows, response_rows, mask_rows = (list(value.unbind()) for value in (prompts, responses, mask))
    if not len(rows) == len(prompt_rows) == len(response_rows) == len(mask_rows):
        raise ValueError("Actor sample counts differ")
    full_masks = []
    for tokens, prompt, response, response_mask in zip(rows, prompt_rows, response_rows, mask_rows, strict=True):
        if prompt.numel() < 1 or response.numel() != response_mask.numel():
            raise ValueError("Invalid prompt or response mask length")
        if not torch.equal(tokens, torch.cat((prompt, response))):
            raise ValueError("input_ids do not match this sample's prompt/response")
        full_masks.append(torch.cat((torch.zeros_like(prompt), response_mask)))
    positions = data.get("position_ids")
    if positions is not None:
        if not positions.is_nested:
            raise ValueError("position_ids must retain jagged sample boundaries")
        position_rows = list(positions.unbind())
        if len(position_rows) != len(rows):
            raise ValueError("Position sample count differs")
        for tokens, position in zip(rows, position_rows, strict=True):
            expected = torch.arange(tokens.numel(), device=position.device)
            if not torch.equal(position, expected):
                raise ValueError("Aligned Nemotron requires positions restarting at zero")
    values = torch.cat(rows)
    return PackedBatch(
        input_ids=values,
        labels=values.clone(),
        seq_lens=torch.tensor([row.numel() for row in rows], dtype=torch.int32, device=values.device),
        loss_mask=torch.cat(full_masks).float(),
    )


def unpack_logprobs(scores: torch.Tensor, batch: PackedBatch) -> torch.Tensor:
    """Restore sample boundaries; verl slices next-token response scores itself."""
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 1 or scores.numel() != batch.total_tokens:
        raise ValueError("Runtime logprob shape does not match the packed batch")
    return torch.nested.nested_tensor_from_jagged(scores, offsets=batch.cu_seqlens.to(torch.int64))
