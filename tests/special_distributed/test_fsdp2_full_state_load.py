"""Regression test for FSDP2 loading from a rank-zero CPU full state.

Launch:
    torchrun --standalone --nproc-per-node=2 \
        tests/special_distributed/test_fsdp2_full_state_load.py
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2, fsdp2_load_full_state_dict


class ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


class BufferedModel(nn.Module):
    _no_split_modules = ["ToyBlock"]

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.block = ToyBlock()
        self.register_buffer("marker", torch.arange(4, dtype=torch.bfloat16), persistent=False)


def _build_model(rank: int) -> tuple[BufferedModel, dict[str, torch.Tensor]]:
    model = BufferedModel()
    state = model.state_dict() if rank == 0 else {}
    return model, state


def _forbid_full_model_to(*args, **kwargs) -> None:
    raise AssertionError("full model must not be materialized with Module.to")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("test_fsdp2_full_state_load skipped: requires two CUDA devices")
        return
    _, rank, world_size = initialize_global_process_group()
    if world_size != 2:
        raise RuntimeError(f"expected two ranks, got {world_size}")
    torch.cuda.set_device(rank)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    model, full_state = _build_model(rank)
    expected = (
        full_state["block.linear.weight"].detach().clone().to("cuda")
        if rank == 0
        else torch.empty((4, 4), device="cuda", dtype=torch.float32)
    )
    dist.broadcast(expected, src=0)
    buffers = {name: buffer.detach().cpu() for name, buffer in model.named_buffers()}
    model.to_empty(device="meta")
    apply_fsdp2(
        model,
        {
            "mesh": mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            ),
            "offload_policy": None,
            "reshard_after_forward": True,
        },
        {"wrap_policy": {"transformer_layer_cls_to_wrap": ["ToyBlock"]}},
    )
    model.to = _forbid_full_model_to
    fsdp2_load_full_state_dict(model, full_state, mesh, buffers=buffers)

    torch.testing.assert_close(model.block.linear.weight.full_tensor().float(), expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(model.marker, torch.arange(4, device="cuda", dtype=torch.bfloat16))
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("test_fsdp2_full_state_load passed")


if __name__ == "__main__":
    main()
