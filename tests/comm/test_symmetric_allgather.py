"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _configure_source_jit_paths() -> None:
    from flashinfer.jit import env as jit_env

    if (jit_env.FLASHINFER_CSRC_DIR / "symmetric_all_gather.cu").exists():
        return
    root = Path(__file__).resolve().parents[2]
    jit_env.FLASHINFER_CSRC_DIR = root / "csrc"
    jit_env.FLASHINFER_INCLUDE_DIR = root / "include"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _expected(
    world_size: int,
    shape: tuple[int, ...],
    offset: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.full(
                shape,
                rank + offset,
                dtype=dtype,
                device="cuda",
            )
            for rank in range(world_size)
        ],
        dim=0,
    )


def _run_worker(
    rank: int,
    world_size: int,
    port: int,
    dtype_name: str,
    shape: tuple[int, ...],
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    workspace = None
    try:
        _configure_source_jit_paths()
        from flashinfer.comm import SymmetricAllGatherWorkspace
        from flashinfer.comm.mnnvl import TorchDistBackend

        backend = TorchDistBackend()
        dtype = getattr(torch, dtype_name)
        elems = math.prod(shape)
        workspace = SymmetricAllGatherWorkspace(
            max_elems=elems + 17,
            world_size=world_size,
            rank=rank,
            comm_backend=backend,
            dtype=dtype,
        )
        eager_input = torch.full(shape, rank + 1, dtype=dtype, device="cuda")
        eager_output = workspace.all_gather(eager_input)
        torch.cuda.synchronize()
        torch.testing.assert_close(eager_output, _expected(world_size, shape, 1, dtype))

        communication_stream = torch.cuda.Stream(device=rank)
        static_input = torch.full(shape, rank + 4, dtype=dtype, device="cuda")
        static_output = torch.empty(
            (shape[0] * world_size, *shape[1:]), dtype=dtype, device="cuda"
        )
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph, stream=communication_stream):
            workspace.all_gather(static_input, static_output)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_output, _expected(world_size, shape, 4, dtype)
        )

        dist.barrier()
        workspace.checkpoint_prepare()
        workspace.checkpoint_prepare()
        with pytest.raises(RuntimeError, match="must be attached"):
            workspace.all_gather(static_input, static_output)
        dist.barrier()
        fresh_backend = TorchDistBackend()
        workspace.checkpoint_restore(fresh_backend)
        workspace.checkpoint_restore(fresh_backend)
        dist.barrier()
        static_input.fill_(rank + 8)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_output, _expected(world_size, shape, 8, dtype)
        )
    finally:
        if workspace is not None:
            workspace.destroy()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="symmetric all-gather requires two CUDA devices",
)
@pytest.mark.parametrize(
    "dtype_name,shape",
    [
        ("float16", (7, 17)),
        ("bfloat16", (16, 256)),
        ("float32", (13, 19)),
    ],
)
def test_graph_replay_after_checkpoint_restore(
    dtype_name: str, shape: tuple[int, ...]
) -> None:
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/flashinfer-tests")
    _configure_source_jit_paths()
    from flashinfer.comm.allgather import _get_module

    _get_module()
    mp.spawn(
        _run_worker,
        args=(2, _free_port(), dtype_name, shape),
        nprocs=2,
        join=True,
    )
