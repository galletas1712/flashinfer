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

CUDA-graph-safe all-gather over peer-mapped symmetric memory.
"""

from __future__ import annotations

from functools import cache
from typing import Optional

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit.comm import gen_symmetric_all_gather_module

from .mnnvl import CommBackend, SymmDeviceMemory


@cache
def _get_module():
    return gen_symmetric_all_gather_module().build_and_load()


class SymmetricAllGatherWorkspace:
    """Persistent peer-mapped workspace for a fixed-size all-gather group.

    All ranks must call all-gather and checkpoint lifecycle operations in the
    same order. Inputs must have the same shape on every rank. Calls submitted
    on different CUDA streams must be externally serialized in that order.
    Repeated successful lifecycle calls are no-ops.
    """

    _DTYPE_CODES = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
    }

    def __init__(
        self,
        max_elems: int,
        world_size: int,
        rank: int,
        comm_backend: CommBackend,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if max_elems <= 0:
            raise ValueError("max_elems must be positive")
        if world_size <= 0 or world_size > 16:
            raise ValueError("world_size must be in [1, 16]")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        if dtype not in self._DTYPE_CODES:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if comm_backend.Get_rank() != rank:
            raise ValueError("comm_backend rank does not match rank")
        if comm_backend.Get_size() != world_size:
            raise ValueError("comm_backend size does not match world_size")

        self.max_elems = max_elems
        self.world_size = world_size
        self.rank = rank
        self.dtype = dtype
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._module = _get_module()
        self._dtype_code = self._DTYPE_CODES[dtype]
        self._element_size = torch.empty((), dtype=dtype).element_size()
        workspace_bytes = int(
            self._module.get_workspace_bytes(max_elems, world_size, self._element_size)
        )
        self._memory = SymmDeviceMemory(
            buf_size=workspace_bytes,
            group_size=world_size,
            group_rank=rank,
            device_idx=self.device.index,
            comm_backend_for_handle_transfer=comm_backend,
            enable_multicast=False,
            allocate_signal_pads=False,
        )
        self._destroyed = False
        self._peer_ptrs = tuple(int(ptr) for ptr in self._memory.get_buffer_ptrs_host())
        self._eager_tickets: dict[int, torch.Tensor] = {}
        self._capture_tickets: list[torch.Tensor] = []
        self._initialize_protocol()
        comm_backend.barrier()

    def all_gather(
        self,
        input: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather equal-shaped inputs in rank-major, dim-0-concatenated order."""
        self._check_attached()
        self._validate_input(input)
        if output is None:
            output = torch.empty(
                (input.shape[0] * self.world_size, *input.shape[1:]),
                dtype=input.dtype,
                device=input.device,
            )
        self._validate_output(input, output)

        stream = torch.cuda.current_stream(self.device).cuda_stream
        if torch.cuda.is_current_stream_capturing():
            ticket = torch.empty(1, dtype=torch.uint64, device=self.device)
            self._capture_tickets.append(ticket)
        else:
            ticket = self._eager_tickets.get(stream)
            if ticket is None:
                ticket = torch.empty(1, dtype=torch.uint64, device=self.device)
                self._eager_tickets[stream] = ticket

        self._module.reserve_sequence(self._peer_ptrs[self.rank], ticket, stream)
        self._module.launch_all_gather(
            self._peer_ptrs[self.rank],
            list(self._peer_ptrs),
            input,
            output,
            input.numel(),
            self.max_elems,
            self.world_size,
            self.rank,
            self._dtype_code,
            ticket,
            stream,
        )
        return output

    @flashinfer_api
    def checkpoint_prepare(self) -> None:
        """Collectively release physical backing while retaining graph addresses."""
        self._check_open()
        if not isinstance(self._memory, SymmDeviceMemory):
            raise NotImplementedError(
                "Stable-VA checkpointing is unavailable for workspaces backed "
                "by torch symmetric memory"
            )
        if not self._memory.mapped:
            return
        self._memory._unmap_and_release_handles()
        # Do not return until every rank has released its workspace handles.
        self._memory.comm_backend.barrier()

    @flashinfer_api
    def checkpoint_restore(self, comm_backend: CommBackend) -> None:
        """Collectively restore physical backing at the original addresses."""
        self._check_open()
        if not isinstance(self._memory, SymmDeviceMemory):
            raise NotImplementedError(
                "Stable-VA checkpointing is unavailable for workspaces backed "
                "by torch symmetric memory"
            )
        if self._memory.mapped:
            return
        self._memory._create_and_map_handles(comm_backend)
        self._initialize_protocol()
        comm_backend.barrier()

    def destroy(self) -> None:
        """Collectively release the workspace."""
        if self._destroyed:
            return
        if self._memory.mapped:
            self.checkpoint_prepare()
        self._capture_tickets.clear()
        self._eager_tickets.clear()
        del self._memory
        self._destroyed = True

    def _initialize_protocol(self) -> None:
        self._module.initialize_workspace(
            self._peer_ptrs[self.rank],
            self.max_elems,
            self.world_size,
            self._element_size,
            self.device.index,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        torch.cuda.synchronize(self.device)

    def _check_open(self) -> None:
        if self._destroyed:
            raise RuntimeError("all-gather workspace is closed")

    def _check_attached(self) -> None:
        self._check_open()
        if not self._memory.mapped:
            raise RuntimeError("all-gather workspace must be attached")

    def _validate_input(self, input: torch.Tensor) -> None:
        if (
            input.dim() == 0
            or input.dtype != self.dtype
            or input.device != self.device
            or not input.is_contiguous()
            or input.numel() <= 0
            or input.numel() > self.max_elems
        ):
            raise ValueError(
                "input must be a non-empty contiguous tensor with the workspace "
                f"dtype and at most {self.max_elems} elements on {self.device}"
            )

    def _validate_output(self, input: torch.Tensor, output: torch.Tensor) -> None:
        expected_shape = (input.shape[0] * self.world_size, *input.shape[1:])
        if (
            output.dtype != input.dtype
            or output.device != input.device
            or not output.is_contiguous()
            or tuple(output.shape) != expected_shape
        ):
            raise ValueError(
                "output must be contiguous, match input dtype/device, and have "
                f"shape {expected_shape}"
            )


@flashinfer_api
def symmetric_all_gather(
    input: torch.Tensor,
    workspace: SymmetricAllGatherWorkspace,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a symmetric-memory all-gather-into-tensor operation."""
    return workspace.all_gather(input, output)
