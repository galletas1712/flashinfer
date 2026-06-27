/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Tensor;

namespace {

constexpr int kNumBuffers = 3;
constexpr int kMaxWorldSize = 16;
constexpr int kThreads = 256;

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct Layout {
  size_t slot_bytes;
  size_t control;
  size_t scratch;
  size_t flags;
  size_t done;
  size_t bytes;
};

struct alignas(8) SequenceControl {
  uint64_t next_sequence;
};

struct alignas(8) SequenceTicket {
  uint64_t sequence;
};

Layout make_layout(size_t elems, int world_size, size_t element_size) {
  TVM_FFI_ICHECK_GT(elems, 0);
  TVM_FFI_ICHECK_GT(world_size, 0);
  TVM_FFI_ICHECK_LE(world_size, kMaxWorldSize);
  TVM_FFI_ICHECK(element_size == 2 || element_size == 4);

  Layout layout{};
  layout.slot_bytes = align_up(elems * element_size, 128);
  size_t offset = 0;
  layout.control = offset;
  offset += sizeof(SequenceControl);
  offset = align_up(offset, 128);
  layout.scratch = offset;
  offset += kNumBuffers * static_cast<size_t>(world_size) * layout.slot_bytes;
  offset = align_up(offset, 128);
  layout.flags = offset;
  offset += kNumBuffers * static_cast<size_t>(world_size) * sizeof(uint64_t);
  offset = align_up(offset, 128);
  layout.done = offset;
  offset += kNumBuffers * static_cast<size_t>(world_size) * sizeof(uint64_t);
  layout.bytes = align_up(offset, 4096);
  return layout;
}

void check_cuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

struct KernelContext {
  char* local_base;
  const char* input;
  char* output;
  char* peer_bases[kMaxWorldSize];
  size_t elems;
  size_t element_size;
  size_t slot_bytes;
  size_t scratch;
  size_t flags;
  size_t done;
  int world_size;
  int rank;
  unsigned long long timeout_cycles;
};

__device__ inline void store_flag_release(uint64_t* address, uint64_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.global.release.sys.b64 [%1], %0;" : : "l"(value), "l"(address));
#else
  __threadfence_system();
  asm volatile("st.global.volatile.b64 [%1], %0;" : : "l"(value), "l"(address));
#endif
}

__device__ inline uint64_t load_flag_acquire(uint64_t* address) {
  uint64_t value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(value) : "l"(address));
#else
  asm volatile("ld.global.volatile.b64 %0, [%1];" : "=l"(value) : "l"(address));
#endif
  return value;
}

__device__ void wait_until_at_least(uint64_t* value, uint64_t expected,
                                    unsigned long long timeout_cycles) {
  const unsigned long long start = clock64();
  while (load_flag_acquire(value) < expected) {
    if (clock64() - start > timeout_cycles) {
      // A timed-out collective cannot be retried safely.
      __trap();
    }
  }
}

__device__ inline void copy_bytes(const char* source, char* destination, size_t bytes, size_t tid,
                                  size_t stride) {
  const uintptr_t alignment =
      reinterpret_cast<uintptr_t>(source) | reinterpret_cast<uintptr_t>(destination) | bytes;
  if ((alignment & (alignof(uint4) - 1)) == 0) {
    const uint4* source_vec = reinterpret_cast<const uint4*>(source);
    uint4* destination_vec = reinterpret_cast<uint4*>(destination);
    const size_t vector_count = bytes / sizeof(uint4);
    for (size_t index = tid; index < vector_count; index += stride) {
      destination_vec[index] = source_vec[index];
    }
    return;
  }
  for (size_t index = tid; index < bytes; index += stride) {
    destination[index] = source[index];
  }
}

__device__ void wait_for_all(uint64_t* values, uint64_t sequence, KernelContext context) {
  for (int source = 0; source < context.world_size; ++source) {
    wait_until_at_least(values + source, sequence, context.timeout_cycles);
  }
}

__device__ void publish_sequence(KernelContext context, size_t offset, uint64_t sequence) {
  __threadfence_system();
  for (int destination = 0; destination < context.world_size; ++destination) {
    if (destination == context.rank) continue;
    store_flag_release(reinterpret_cast<uint64_t*>(context.peer_bases[destination] + offset),
                       sequence);
  }
  store_flag_release(reinterpret_cast<uint64_t*>(context.local_base + offset), sequence);
}

__device__ void all_gather_step(KernelContext context, uint64_t sequence) {
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  const size_t buffer = sequence % kNumBuffers;

  uint64_t* local_done = reinterpret_cast<uint64_t*>(
      context.local_base + context.done + buffer * context.world_size * sizeof(uint64_t));
  if (threadIdx.x == 0 && sequence > kNumBuffers) {
    wait_for_all(local_done, sequence - kNumBuffers, context);
  }
  __syncthreads();

  const size_t payload_bytes = context.elems * context.element_size;
  const size_t target_offset =
      context.scratch + (buffer * context.world_size + context.rank) * context.slot_bytes;
  for (int destination = 0; destination < context.world_size; ++destination) {
    copy_bytes(context.input, context.peer_bases[destination] + target_offset, payload_bytes, tid,
               stride);
  }

  __threadfence_system();
  __syncthreads();
  uint64_t* local_flags = reinterpret_cast<uint64_t*>(
      context.local_base + context.flags + buffer * context.world_size * sizeof(uint64_t));
  if (threadIdx.x == 0) {
    const size_t flag_offset =
        context.flags + (buffer * context.world_size + context.rank) * sizeof(uint64_t);
    publish_sequence(context, flag_offset, sequence);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    wait_for_all(local_flags, sequence, context);
  }
  __syncthreads();

  for (int source = 0; source < context.world_size; ++source) {
    const char* source_slot = context.local_base + context.scratch +
                              (buffer * context.world_size + source) * context.slot_bytes;
    copy_bytes(source_slot, context.output + static_cast<size_t>(source) * payload_bytes,
               payload_bytes, tid, stride);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    const size_t done_offset =
        context.done + (buffer * context.world_size + context.rank) * sizeof(uint64_t);
    publish_sequence(context, done_offset, sequence);
  }
  __syncthreads();
}

__global__ void all_gather_kernel(KernelContext context, const SequenceTicket* ticket) {
  const SequenceTicket reservation = *ticket;
  if (reservation.sequence == 0) {
    // Sequence wrap makes the captured protocol invalid.
    if (threadIdx.x == 0) __trap();
    return;
  }
  all_gather_step(context, reservation.sequence);
}

__global__ void initialize_control_kernel(SequenceControl* control) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    control->next_sequence = 1;
  }
}

__global__ void reserve_sequence_kernel(SequenceControl* control, SequenceTicket* ticket) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ticket->sequence =
        atomicAdd(reinterpret_cast<unsigned long long*>(&control->next_sequence), 1ULL);
  }
}

int64_t get_workspace_bytes(int64_t max_elems, int64_t world_size, int64_t element_size) {
  return static_cast<int64_t>(make_layout(static_cast<size_t>(max_elems),
                                          static_cast<int>(world_size),
                                          static_cast<size_t>(element_size))
                                  .bytes);
}

void initialize_workspace(int64_t local_ptr, int64_t max_elems, int64_t world_size,
                          int64_t element_size, int64_t device, int64_t stream_ptr) {
  TVM_FFI_ICHECK_NE(local_ptr, 0);
  ffi::CUDADeviceGuard guard(static_cast<int>(device));
  Layout layout = make_layout(static_cast<size_t>(max_elems), static_cast<int>(world_size),
                              static_cast<size_t>(element_size));
  char* local_base = reinterpret_cast<char*>(static_cast<uintptr_t>(local_ptr));
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  check_cuda(cudaMemsetAsync(local_base, 0, layout.bytes, stream),
             "cudaMemsetAsync(all-gather workspace)");
  initialize_control_kernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<SequenceControl*>(local_base + layout.control));
  check_cuda(cudaGetLastError(), "initialize_control_kernel");
}

void reserve_sequence(int64_t local_ptr, Tensor ticket, int64_t stream_ptr) {
  TVM_FFI_ICHECK_NE(local_ptr, 0);
  CHECK_INPUT_AND_TYPE(ticket, dl_uint64);
  TVM_FFI_ICHECK_GE(ticket.numel(), 1);
  ffi::CUDADeviceGuard guard(ticket.device().device_id);
  char* local_base = reinterpret_cast<char*>(static_cast<uintptr_t>(local_ptr));
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  reserve_sequence_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<SequenceControl*>(local_base),
                                               static_cast<SequenceTicket*>(ticket.data_ptr()));
  check_cuda(cudaGetLastError(), "reserve_sequence_kernel");
}

void launch_all_gather(int64_t local_ptr, Array<int64_t> peer_ptrs, Tensor input, Tensor output,
                       int64_t elems, int64_t max_elems, int64_t world_size, int64_t rank,
                       int64_t dtype, Tensor ticket, int64_t stream_ptr) {
  TVM_FFI_ICHECK_NE(local_ptr, 0);
  TVM_FFI_ICHECK_GT(elems, 0);
  TVM_FFI_ICHECK_LE(elems, max_elems);
  TVM_FFI_ICHECK_EQ(peer_ptrs.size(), static_cast<size_t>(world_size));
  TVM_FFI_ICHECK_GE(rank, 0);
  TVM_FFI_ICHECK_LT(rank, world_size);
  TVM_FFI_ICHECK_EQ(input.numel(), elems);
  TVM_FFI_ICHECK_EQ(output.numel(), elems * world_size);
  TVM_FFI_ICHECK_EQ(input.device().device_id, output.device().device_id);
  CHECK_INPUT_AND_TYPE(ticket, dl_uint64);
  TVM_FFI_ICHECK_GE(ticket.numel(), 1);
  TVM_FFI_ICHECK_EQ(ticket.device().device_id, input.device().device_id);

  size_t element_size = 0;
  if (dtype == 0) {
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(input.dtype()), float16_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), float16_code);
    element_size = 2;
  } else if (dtype == 1) {
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(input.dtype()), bfloat16_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), bfloat16_code);
    element_size = 2;
  } else if (dtype == 2) {
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(input.dtype()), float32_code);
    TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(output.dtype()), float32_code);
    element_size = 4;
  } else {
    TVM_FFI_ICHECK(false) << "unsupported dtype code " << dtype;
  }

  Layout layout =
      make_layout(static_cast<size_t>(max_elems), static_cast<int>(world_size), element_size);
  KernelContext context{};
  context.local_base = reinterpret_cast<char*>(static_cast<uintptr_t>(local_ptr));
  context.input = static_cast<const char*>(input.data_ptr());
  context.output = static_cast<char*>(output.data_ptr());
  for (int peer = 0; peer < world_size; ++peer) {
    TVM_FFI_ICHECK_NE(peer_ptrs[peer], 0);
    context.peer_bases[peer] = reinterpret_cast<char*>(static_cast<uintptr_t>(peer_ptrs[peer]));
  }
  TVM_FFI_ICHECK_EQ(context.peer_bases[rank], context.local_base);
  context.elems = static_cast<size_t>(elems);
  context.element_size = element_size;
  context.slot_bytes = layout.slot_bytes;
  context.scratch = layout.scratch;
  context.flags = layout.flags;
  context.done = layout.done;
  context.world_size = static_cast<int>(world_size);
  context.rank = static_cast<int>(rank);
  context.timeout_cycles = 30000000000ULL;

  ffi::CUDADeviceGuard guard(input.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  all_gather_kernel<<<1, kThreads, 0, stream>>>(
      context, static_cast<const SequenceTicket*>(ticket.data_ptr()));
  check_cuda(cudaGetLastError(), "all_gather_kernel");
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_workspace_bytes, get_workspace_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(initialize_workspace, initialize_workspace);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reserve_sequence, reserve_sequence);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch_all_gather, launch_all_gather);
