# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from flashinfer.comm.allreduce import TRTLLMAllReduceFusionWorkspace
from flashinfer.comm.trtllm_mnnvl_ar import MNNVLAllReduceFusionWorkspace


class _FakeComm:
    def __init__(self):
        self.barriers = 0

    def barrier(self):
        self.barriers += 1


class _FakeCheckpointableHandle:
    def __init__(self):
        self.comm_backend = _FakeComm()
        self.calls = []

    def validate_graph_visible_addresses(self):
        self.calls.append(("validate",))

    def detach_physical_keep_va(self, *, synchronize=True, barrier=True):
        self.calls.append(("detach", synchronize, barrier))

    def remap_physical_same_va(
        self, *, comm_backend=None, synchronize=True, barrier=True, zero_local=True
    ):
        self.calls.append(("remap", comm_backend, synchronize, barrier, zero_local))

    def lamport_initialize(self, rank, dtype):
        self.calls.append(("lamport", rank, dtype))


def _make_trtllm_workspace():
    workspace = object.__new__(TRTLLMAllReduceFusionWorkspace)
    workspace.world_size = 2
    workspace.rank = 0
    workspace.ipc_handles = [[0x1000, 0x2000], [0x3000, 0x4000], [0x5000, 0x6000]]
    workspace.workspace_tensor = torch.tensor(
        [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000],
        dtype=torch.int64,
    )
    workspace.mem_handles = [_FakeCheckpointableHandle() for _ in range(3)]
    workspace.metadata = {
        "use_fp32_lamport": False,
        "lamport_comm_size": 1024,
    }
    workspace._graph_visible_addresses = workspace.get_graph_visible_addresses()
    workspace._destroyed = False
    return workspace


def test_trtllm_allreduce_checkpoint_hooks_delegate_to_handles():
    workspace = _make_trtllm_workspace()

    workspace.detach_physical_keep_va(synchronize=False, barrier=False)
    workspace.remap_physical_same_va(
        comm_backend="fresh-comm",
        synchronize=False,
        barrier=False,
        reset=False,
    )

    for handle in workspace.mem_handles:
        assert ("validate",) in handle.calls
        assert ("detach", False, False) in handle.calls
        assert ("remap", "fresh-comm", False, False, True) in handle.calls


def test_trtllm_allreduce_checkpoint_hooks_fail_closed_without_handle_support():
    workspace = _make_trtllm_workspace()
    workspace.mem_handles = [object()]

    try:
        workspace.detach_physical_keep_va()
    except RuntimeError as exc:
        assert "does not support" in str(exc)
    else:
        raise AssertionError("checkpoint pause should fail without handle support")


def test_mnnvl_allreduce_checkpoint_hooks_delegate_to_allocator_handle():
    workspace = object.__new__(MNNVLAllReduceFusionWorkspace)
    workspace.world_size = 2
    workspace.rank = 0
    workspace.tp_size = 2
    workspace.ptrs = [0x1000, 0x2000]
    workspace.uc_ptrs_dev = 0x3000
    workspace.uc_ptr_local = 0x1000
    workspace.mc_ptr = 0x4000
    workspace.buffer_size_bytes = 128
    workspace.workspace_size_bytes = 384
    workspace.handle = _FakeCheckpointableHandle()
    workspace.comm_backend = workspace.handle.comm_backend
    workspace._graph_visible_addresses = workspace.get_graph_visible_addresses()

    workspace.detach_physical_keep_va(synchronize=False, barrier=False)
    workspace.remap_physical_same_va(
        comm_backend="fresh-comm",
        synchronize=False,
        barrier=False,
        reset=False,
    )

    assert ("detach", False, False) in workspace.handle.calls
    assert ("remap", "fresh-comm", False, False, True) in workspace.handle.calls
