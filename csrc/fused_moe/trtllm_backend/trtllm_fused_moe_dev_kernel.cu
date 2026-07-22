/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function for array conversion
template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
  return converter(input);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace moe::dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace activation {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float silu(float x) { return x / (1.0f + expf(-x)); }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void activationKernel(KernelParams params) {
  using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.z; tokenIdx < params.numTokens; tokenIdx += gridDim.z) {
    // Look over experts per token
    for (int k = blockIdx.y; k < params.topK; k += gridDim.y) {
      int const expandedIdx = tokenIdx * params.topK + k;
      int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
      if (permutedIdx == -1) continue;

      // Loop over hidden dim
      for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.innerDim / 2;
           hiddenIdx += blockDim.x * gridDim.x) {
        // Use int64_t to avoid overflow when permutedIdx * innerDim > INT32_MAX
        int64_t const baseIdx = (int64_t)permutedIdx * params.innerDim + hiddenIdx;

        float x1 = (float)params.inPtr[baseIdx];
        float x2 = (float)params.inPtr[baseIdx + params.innerDim / 2];

        float act = silu(x2);
        Type out = (Type)(act * x1);

        int64_t const outIdx = (int64_t)permutedIdx * (params.innerDim / 2) + hiddenIdx;
        params.outPtr[outIdx] = out;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Float4Max {
  __device__ __forceinline__ float4 operator()(float4 const& a, float4 const& b) const {
    float4 result;
    result.x = fmaxf(a.x, b.x);
    result.y = fmaxf(a.y, b.y);
    result.z = fmaxf(a.z, b.z);
    result.w = fmaxf(a.w, b.w);
    return result;
  }
};

struct Float2Max {
  __device__ __forceinline__ float2 operator()(float2 const& a, float2 const& b) const {
    float2 result;
    result.x = fmaxf(a.x, b.x);
    result.y = fmaxf(a.y, b.y);
    return result;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VecType, int size>
__device__ __forceinline__ VecType packedTypeFromArray(float data[size]) {
  return {};
}

template <>
__device__ __forceinline__ float4 packedTypeFromArray<float4, 4>(float data[4]) {
  float4 result;
  result.x = data[0];
  result.y = data[1];
  result.z = data[2];
  result.w = data[3];
  return result;
}

template <>
__device__ __forceinline__ float2 packedTypeFromArray<float2, 2>(float data[2]) {
  float2 result;
  result.x = data[0];
  result.y = data[1];
  return result;
}

template <>
__device__ __forceinline__ float packedTypeFromArray<float, 1>(float data[1]) {
  return data[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename PackedType, int size>
__device__ __forceinline__ cutlass::Array<float, size> arrayFromPackedType(PackedType data) {
  return cutlass::Array<float, size>{};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 4> arrayFromPackedType<float4, 4>(float4 data) {
  return cutlass::Array<float, 4>{data.x, data.y, data.z, data.w};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 2> arrayFromPackedType<float2, 2>(float2 data) {
  return cutlass::Array<float, 2>{data.x, data.y};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 1> arrayFromPackedType<float, 1>(float data) {
  return cutlass::Array<float, 1>{data};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NUM_TOKENS_PER_CTA>
struct KernelTraits;

template <>
struct KernelTraits<4> {
  using MaxOp = Float4Max;
  using PackedType = float4;
};

template <>
struct KernelTraits<2> {
  using MaxOp = Float2Max;
  using PackedType = float2;
};

template <>
struct KernelTraits<1> {
  using MaxOp = cuda::maximum<>;
  using PackedType = float;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA = 128;

template <typename KernelParams>
__global__ void activationDeepSeekKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  int32_t constexpr NumTokensPerCta = KernelParams::NumTokensPerCta;
  using KernelTraits = KernelTraits<NumTokensPerCta>;
  using MaxOp = typename KernelTraits::MaxOp;
  using PackedType = typename KernelTraits::PackedType;
  using BlockReduce = cub::BlockReduce<PackedType, DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA>;

  __shared__ float s_scaleOutArr[NumTokensPerCta];
  __shared__ typename BlockReduce::TempStorage tempStorage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  // The largest (finite) value that can be represented using E4m3.
  float constexpr E4m3MaxVal{448.f};

  int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
  // Loop over tokens
  float scale1Arr[NumTokensPerCta];
  float scale2Arr[NumTokensPerCta];
  float dataX1Arr[NumTokensPerCta];
  float dataX2Arr[NumTokensPerCta];
  float outArr[NumTokensPerCta];
  float absOutArr[NumTokensPerCta];
  int permutedIdxArr[NumTokensPerCta];

  // Loop over tokens
  for (int k = blockIdx.z; k < params.topK; k += gridDim.z) {
    for (int tokenCtaIdx = blockIdx.y * NumTokensPerCta; tokenCtaIdx < params.numTokens;
         tokenCtaIdx += gridDim.y * NumTokensPerCta) {
      for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.innerDim / 2;
           hiddenIdx += blockDim.x * gridDim.x) {
#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
          scale1Arr[tokenInCtaIdx] = 0.0f;
          scale2Arr[tokenInCtaIdx] = 0.0f;
          dataX1Arr[tokenInCtaIdx] = 0.0f;
          dataX2Arr[tokenInCtaIdx] = 0.0f;
          outArr[tokenInCtaIdx] = 0.0f;
          absOutArr[tokenInCtaIdx] = 0.0f;
        }
#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
          int const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
          if (tokenIdx >= params.numTokens) {
            break;
          }

          int const expandedIdx = tokenIdx * params.topK + k;
          int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
          permutedIdxArr[tokenInCtaIdx] = permutedIdx;
          if (permutedIdx == -1) {
            continue;
          }

          // Process blocks for this CTA
          // Use int64_t to avoid overflow when permutedIdx * innerDim > INT32_MAX
          int64_t const baseIdx = (int64_t)permutedIdx * params.innerDim + hiddenIdx;

          int64_t const scale1Idx =
              (int64_t)permutedIdx + (int64_t)totalNumPaddedTokens * (hiddenIdx / 128);
          int64_t const scale2Idx =
              (int64_t)permutedIdx +
              (int64_t)totalNumPaddedTokens * ((hiddenIdx / 128) + (params.innerDim / 2 / 128));

          scale1Arr[tokenInCtaIdx] = params.inDqSfsPtr[scale1Idx];
          scale2Arr[tokenInCtaIdx] = params.inDqSfsPtr[scale2Idx];
          dataX1Arr[tokenInCtaIdx] = static_cast<float>(params.inPtr[baseIdx]);
          dataX2Arr[tokenInCtaIdx] =
              static_cast<float>(params.inPtr[baseIdx + params.innerDim / 2]);
        }

#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
          float x1 = scale1Arr[tokenInCtaIdx] * dataX1Arr[tokenInCtaIdx];
          float x2 = scale2Arr[tokenInCtaIdx] * dataX2Arr[tokenInCtaIdx];
          float act = silu(x2);
          float out = act * x1;
          outArr[tokenInCtaIdx] = out;
          absOutArr[tokenInCtaIdx] = fabsf(out);
        }

        auto absOutPacked = packedTypeFromArray<PackedType, NumTokensPerCta>(absOutArr);
        auto aMaxPacked = BlockReduce(tempStorage).Reduce(absOutPacked, MaxOp{});
        auto aMaxArr = arrayFromPackedType<PackedType, NumTokensPerCta>(aMaxPacked);

#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
          if (threadIdx.x == 0) {
            auto const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
            if (tokenIdx >= params.numTokens) {
              break;
            }
            int const permutedIdx = permutedIdxArr[tokenInCtaIdx];
            if (permutedIdx == -1) {
              continue;
            }
            // Make sure the scale is strictly positive to avoid division by zero in case the
            // maximum is zero.
            float scaleOut =
                fmaxf(aMaxArr[tokenInCtaIdx] / E4m3MaxVal, std::numeric_limits<float>::min());
            s_scaleOutArr[tokenInCtaIdx] = scaleOut;
            int64_t const scaleOut_idx = (int64_t)permutedIdxArr[tokenInCtaIdx] +
                                         (int64_t)totalNumPaddedTokens * (hiddenIdx / 128);
            params.outDqSfsPtr[scaleOut_idx] = scaleOut;
          }
        }
        __syncthreads();

#pragma unroll
        for (int tokenInCtaIdx = 0; tokenInCtaIdx < NumTokensPerCta; tokenInCtaIdx++) {
          auto const tokenIdx = tokenCtaIdx + tokenInCtaIdx;
          if (tokenIdx >= params.numTokens) {
            break;
          }
          int const permutedIdx = permutedIdxArr[tokenInCtaIdx];
          if (permutedIdx == -1) {
            continue;
          }
          float const scaleOut = s_scaleOutArr[tokenInCtaIdx];
          int64_t const outIdx = (int64_t)permutedIdx * (params.innerDim / 2) + hiddenIdx;
          params.outPtr[outIdx] = static_cast<Type>(outArr[tokenInCtaIdx] / scaleOut);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  if (data.mDtypeElt == tg::Dtype::E2m1) {
    // Note: this should be unreachable because the options are checked beforehand.
    // E2m1 requires using higher-precision intermediate data (bf16).
    FLASHINFER_CHECK(false, "Activation with E2m1_t isn't supported.");
    return;
  }

  if (data.mUseDeepSeekFp8) {
    constexpr int NUM_ELTS_PER_LOAD = 1;
    constexpr int NUM_ELTS_PER_SF = 128;

    int device{-1};
    cudaGetDevice(&device);
    int numSms = 0;
    cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device);

    // Output dimension is innerDim / 2, and each scale block is 128 elements
    int const outputDim = data.innerDim / 2;
    int const numScaleBlocks = (outputDim + NUM_ELTS_PER_SF - 1) / NUM_ELTS_PER_SF;
    int const gridSizeX = (numScaleBlocks + NUM_ELTS_PER_LOAD - 1) / NUM_ELTS_PER_LOAD;

    auto numCtas = gridSizeX * data.numTokens * data.topK;
    // FIXME: This is heruistic based on very short benchmark.
    int numTokensPerCta = 1;
    if (numCtas > numSms * 32) {
      numTokensPerCta = 4;
    } else if (numCtas > numSms * 4) {
      numTokensPerCta = 2;
    } else {
      numTokensPerCta = 1;
    }

    int const gridSizeY = std::min(8192, (data.numTokens + numTokensPerCta - 1) / numTokensPerCta);

    const dim3 grid(gridSizeX, gridSizeY, data.topK);

    LAUNCH_ACTIVATION(data, activationDeepSeekKernel, numTokensPerCta, grid,
                      DEEP_SEEK_ACTIVATION_NUM_THREADS_PER_CTA, 0, stream);
  } else {
    int const numThreads = 256;
    const dim3 grid(data.innerDim / 128, data.topK, std::min(8192, data.numTokens));

    LAUNCH_ACTIVATION(data, activationKernel, 1, grid, numThreads, 0, stream);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace activation

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace convertsf {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

namespace dev {
// Compute the offset that corresponds to (dataRowIdx, dataBlkColIdx) in the SF tensor where
// dataRowIdx and dataBlkColIdx are the respective indices of the row and the block of 16 elts
// from the K dim in the tensor of data.
inline __device__ int64_t getSfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx,
                                      int32_t numDataBlksPerRow) {
  // The number of rows of SF per block.
  static int32_t constexpr NumRowsPerSfBlock = 128;
  // The number of cols of SF per block.
  static int32_t constexpr NumColsPerSfBlock = 4;
  // The size of each SF block.
  static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;

  // The number of rows of data per SF block.
  static int32_t constexpr NumDataRowsPerSfBlock = NumRowsPerSfBlock;
  // The number of cols of blocks of data per SF block.
  static int32_t constexpr NumDataBlkColsPerSfBlock = NumColsPerSfBlock;

  // The row of the SF block in the SF tensor.
  int sfBlkRowIdx = dataRowIdx / NumDataRowsPerSfBlock;
  // The col of the SF block in the SF tensor.
  int sfBlkColIdx = dataBlkColIdx / NumDataBlkColsPerSfBlock;
  // The blocks are stored row-major in the tensor of scaling factors.
  int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / NumDataBlkColsPerSfBlock + sfBlkColIdx;

  // Find the row in the SF block.
  int sfRowIdx = (dataRowIdx % 32) * 4 + (dataRowIdx % NumDataRowsPerSfBlock) / 32;
  // Find the col in the SF block.
  int sfColIdx = (dataBlkColIdx % 4);

  // Compute the offset in bytes.
  return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock + sfColIdx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Given the GMEM address of an output element, compute the offset of the corresponding scaling
// factor in the SF tensor. Optionally, a startTokenIndex can be provided if the first token is not
// the start token in the SF tensor. This is useful when inflight batching is enabled in TRT-LLM,
// where the context and generation output are stored as one output tensor. In this case, the
// generation output may not start with zero offset in the SF output tensor.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(int64_t gmemOffsetInBytes, int32_t hiddenDim,
                                      int32_t startTokenIdx = 0) {
  // The number of elements per sf.
  int32_t constexpr NumEltsPerSf = 16;
  // The GMEM offset of the output element.
  int64_t gmemOffset = gmemOffsetInBytes * 8 /*bits*/ / NumBitsPerElt;
  // The row/col indices of the corresponding SF element.
  int32_t sfRowIdx = gmemOffset / hiddenDim + startTokenIdx;
  int32_t sfColIdx = (gmemOffset % hiddenDim) / NumEltsPerSf;
  // Compute the SF offset.
  return getSfOffset(sfRowIdx, sfColIdx, hiddenDim / NumEltsPerSf);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO(tizheng): Refactor to track gmem offset instead of doing pointer subtraction.
template <int32_t NumBitsPerElt>
inline __device__ int64_t getSfOffset(void const* gmemOutPtr, void const* gmemBasePtr,
                                      int32_t hiddenDim, int32_t startTokenIdx = 0) {
  return getSfOffset<NumBitsPerElt>(
      reinterpret_cast<char const*>(gmemOutPtr) - reinterpret_cast<char const*>(gmemBasePtr),
      hiddenDim, startTokenIdx);
}

}  // namespace dev

// TODO: it would be nice to move some of that logic to Fp4Utils.h
template <tg::SfLayout Layout>
inline __device__ int32_t getSfOffset(int32_t dataRowIdx, int32_t dataBlkColIdx,
                                      int32_t numDataBlksPerRow) {
  if constexpr (Layout == tg::SfLayout::Linear) {
    return numDataBlksPerRow * dataRowIdx + dataBlkColIdx;
  } else if constexpr (Layout == tg::SfLayout::R128c4) {
    return static_cast<int32_t>(dev::getSfOffset(dataRowIdx, dataBlkColIdx, numDataBlksPerRow));
  } else if constexpr (Layout == tg::SfLayout::R8c4 || Layout == tg::SfLayout::R8c16) {
    static int32_t constexpr NumRowsPerSfBlock = 8;
    static int32_t constexpr NumColsPerSfBlock = (Layout == tg::SfLayout::R8c4) ? 4 : 16;
    static int32_t constexpr NumBytesPerSfBlock = NumRowsPerSfBlock * NumColsPerSfBlock;
    int sfBlkRowIdx = dataRowIdx / NumRowsPerSfBlock;
    int sfBlkColIdx = dataBlkColIdx / NumColsPerSfBlock;
    int sfBlkIdx = sfBlkRowIdx * numDataBlksPerRow / NumColsPerSfBlock + sfBlkColIdx;
    int sfRowIdx = dataRowIdx % NumRowsPerSfBlock;
    int sfColIdx = dataBlkColIdx % NumColsPerSfBlock;
    return sfBlkIdx * NumBytesPerSfBlock + sfRowIdx * NumColsPerSfBlock + sfColIdx;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <tg::SfLayout LayoutSrc, tg::SfLayout LayoutDst, typename KernelParams>
__device__ void convertSfCommon(KernelParams params) {
  // Note: it's assumed that the number of scaling factors per row is a multiple of 4.
  constexpr int VecSize = 4;
  using VecType = uint32_t;
  static_assert(sizeof(VecType) == VecSize);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // Immediately trigger the secondary kernel when using PDL, then wait on primary.
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  // TODO: consider optimizing if used in production.
  // This is a naive kernel. It's not doing coalesced loads.

  int const numSfPerRow = params.hiddenDimSf;

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y) {
    for (int hiddenSfVecIdx = threadIdx.x + blockDim.x * blockIdx.x;
         hiddenSfVecIdx < numSfPerRow / VecSize; hiddenSfVecIdx += blockDim.x * gridDim.x) {
      // Index of the first SF in the vector.
      int const hiddenSfIdx = VecSize * hiddenSfVecIdx;

      // Load scale factors.
      int sfIdxIn = getSfOffset<LayoutSrc>(tokenIdx, hiddenSfIdx, numSfPerRow);
      const VecType sfVec = reinterpret_cast<VecType const*>(params.inSfPtr)[sfIdxIn / VecSize];

      // Store scale factors.
      int const sfIdxOut = getSfOffset<LayoutDst>(tokenIdx, hiddenSfIdx, numSfPerRow);
      reinterpret_cast<VecType*>(params.outSfPtr)[sfIdxOut / VecSize] = sfVec;
    }
  }
}

#define CONVERT_FP4_SF_KERNEL(LayoutSrc, LayoutDst)                                  \
  template <typename KernelParams>                                                   \
  __global__ void convertSf##LayoutSrc##To##LayoutDst##Kernel(KernelParams params) { \
    convertSfCommon<tg::SfLayout::LayoutSrc, tg::SfLayout::LayoutDst>(params);       \
  }
// We only need a conversion to the linear layout.
CONVERT_FP4_SF_KERNEL(R128c4, Linear);
CONVERT_FP4_SF_KERNEL(R8c4, Linear);
CONVERT_FP4_SF_KERNEL(R8c16, Linear);
#undef CONVERT_FP4_SF_KERNEL

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  constexpr int VecSize = 4;
  int const numThreads = 128;
  int const numBlocksX = (data.hiddenDimSf / VecSize - 1 + numThreads) / numThreads;
  int const numBlocksY = std::min(8192, data.numTokens);
  dim3 numBlocks(numBlocksX, numBlocksY);
#define CONVERT_FP4_SF_LAUNCH(LayoutSrc, LayoutDst)                                             \
  if (data.sfLayoutSrc == tg::SfLayout::LayoutSrc &&                                            \
      data.sfLayoutDst == tg::SfLayout::LayoutDst) {                                            \
    LAUNCH_PDL(data, false, cutlass::float_e4m3_t, convertSf##LayoutSrc##To##LayoutDst##Kernel, \
               numBlocks, numThreads, 0, stream);                                               \
    return;                                                                                     \
  }
  CONVERT_FP4_SF_LAUNCH(R128c4, Linear);
  CONVERT_FP4_SF_LAUNCH(R8c4, Linear);
  CONVERT_FP4_SF_LAUNCH(R8c16, Linear);
#undef CONVERT_FP4_SF_LAUNCH
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace convertsf

namespace permute {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void permuteKernel(KernelParams params) {
  using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // immediately trigger the secondary kernel when using PDL, then wait on primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y) {
    // Loop over hidden dim
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
         hiddenIdx += blockDim.x * gridDim.x) {
      // Load chunk of token into registers
      const Type data = params.inPtr[tokenIdx * params.hiddenDim + hiddenIdx];

      // Write to topK places
      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
        params.outPtr[permutedIdx * params.hiddenDim + hiddenIdx] = data;
      }
    }
    if (params.useDeepSeekFp8) {
      for (int scaleIdx = threadIdx.x + blockDim.x * blockIdx.x; scaleIdx < params.hiddenDim / 128;
           scaleIdx += blockDim.x * gridDim.x) {
        for (int k = 0; k < params.topK; k++) {
          int const expandedIdx = tokenIdx * params.topK + k;
          int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

          int const idx_in = tokenIdx + params.numTokens * scaleIdx;
          int const idx_out = permutedIdx + params.totalNumPaddedTokens[0] * scaleIdx;

          params.outDqSfsPtr[idx_out] = params.inDqSfsPtr[idx_in];
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  int const numThreads = 256;
  int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
  int const numBlocksY = std::min(8192, data.numTokens);
  dim3 numBlocks(numBlocksX, numBlocksY);

  LAUNCH(data, permuteKernel, numBlocks, numThreads, 0, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace permute

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace finalize {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = batchedGemm::trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  using TypeExpW = typename KernelParams::TypeExpW;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y) {
    // Loop over hidden dim
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
         hiddenIdx += blockDim.x * gridDim.x) {
      // Accumulate chunk of token into registers
      float data = 0.0F;

      // Write to topK places
      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

        if (permutedIdx == -1) {
          continue;
        }

        if (params.expertWeightsPtr != nullptr) {
          TypeExpW const scale = params.expertWeightsPtr[expandedIdx];
          data +=
              float{scale} * float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
        } else {
          data += float{params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]};
        }
      }

      params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] = static_cast<Type>(data);
    }
  }
}

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

__device__ float4 vectorizedLoadPtx(float4 const* ptr) {
  float4 ret;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
               : "l"(ptr));
  return ret;
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip
// connection.
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int MaxTopK = 64;

typedef struct __CUDA_ALIGN__(4) {
  cutlass::bfloat16_t array[2];
} bfloat16_2;

typedef struct __CUDA_ALIGN__(8) {
  cutlass::bfloat16_t array[4];
} bfloat16_4;

typedef struct __CUDA_ALIGN__(8) {
  half array[4];
} half_4;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int UnrollFactor_, typename TypeExpW_>
struct ScaleTraitsStruct;

template <>
struct ScaleTraitsStruct<1, cutlass::bfloat16_t> {
  using PackedType = cutlass::bfloat16_t;
  using ArrayType = cutlass::Array<cutlass::bfloat16_t, 1>;
};

template <>
struct ScaleTraitsStruct<2, cutlass::bfloat16_t> {
  using PackedType = bfloat16_2;
  using ArrayType = cutlass::Array<cutlass::bfloat16_t, 2>;
};

template <>
struct ScaleTraitsStruct<4, cutlass::bfloat16_t> {
  using PackedType = bfloat16_4;
  using ArrayType = cutlass::Array<cutlass::bfloat16_t, 4>;
};

template <>
struct ScaleTraitsStruct<1, float> {
  using PackedType = float;
  using ArrayType = cutlass::Array<float, 1>;
};

template <>
struct ScaleTraitsStruct<2, float> {
  using PackedType = float2;
  using ArrayType = cutlass::Array<float, 2>;
};

template <>
struct ScaleTraitsStruct<4, float> {
  using PackedType = float4;
  using ArrayType = cutlass::Array<float, 4>;
};

template <>
struct ScaleTraitsStruct<1, half> {
  using PackedType = half;
  using ArrayType = cutlass::Array<half, 1>;
};

template <>
struct ScaleTraitsStruct<2, half> {
  using PackedType = half2;
  using ArrayType = cutlass::Array<half, 2>;
};

template <>
struct ScaleTraitsStruct<4, half> {
  using PackedType = half_4;
  using ArrayType = cutlass::Array<half, 4>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int UnrollFactor_, typename TypeExpW_>
struct FinalizeTraits;

template <typename TypeExpW_>
struct FinalizeTraits<1, TypeExpW_> {
  using IdxPackedType = int;
  using IdxArrayType = cutlass::Array<int, 1>;
  using ScaleTraits = ScaleTraitsStruct<1, TypeExpW_>;
  using ScalePackedType = typename ScaleTraits::PackedType;
  using ScaleArrayType = typename ScaleTraits::ArrayType;
};

template <typename TypeExpW_>
struct FinalizeTraits<2, TypeExpW_> {
  using IdxPackedType = int2;
  using IdxArrayType = cutlass::Array<int, 2>;
  using ScaleTraits = ScaleTraitsStruct<2, TypeExpW_>;
  using ScalePackedType = typename ScaleTraits::PackedType;
  using ScaleArrayType = typename ScaleTraits::ArrayType;
};

template <typename TypeExpW_>
struct FinalizeTraits<4, TypeExpW_> {
  using IdxPackedType = int4;
  using IdxArrayType = cutlass::Array<int, 4>;
  using ScaleTraits = ScaleTraitsStruct<4, TypeExpW_>;
  using ScalePackedType = typename ScaleTraits::PackedType;
  using ScaleArrayType = typename ScaleTraits::ArrayType;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeKernelVecLoad(KernelParams params) {
  using Type = typename KernelParams::Type;
  using TypeExpW = typename KernelParams::TypeExpW;
  int constexpr TopKUnrollFactor = KernelParams::TopKUnrollFactor;

  static_assert(TopKUnrollFactor == 1 || TopKUnrollFactor == 2 || TopKUnrollFactor == 4,
                "TopKUnrollFactor must be 1, 2, or 4");
  using FinalizeTraits = FinalizeTraits<TopKUnrollFactor, TypeExpW>;
  using IdxPackedType = typename FinalizeTraits::IdxPackedType;
  using IdxArrayType = typename FinalizeTraits::IdxArrayType;
  using ScalePackedType = typename FinalizeTraits::ScalePackedType;
  using ScaleArrayType = typename FinalizeTraits::ScaleArrayType;

  int const hiddenDimPaddedBits = params.hiddenDimPadded * cutlass::sizeof_bits<Type>::value;
  int const hiddenDimBits = params.hiddenDim * cutlass::sizeof_bits<Type>::value;
  assert(hiddenDimPaddedBits % 128 == 0);
  assert(hiddenDimBits % 128 == 0);

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<Type>::value;
  using InputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<Type, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;

  int64_t const tokenIdx = blockIdx.x;
  int64_t const startOffset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const numElemsInPaddedCol = params.hiddenDimPadded / FINALIZE_ELEM_PER_THREAD;
  int64_t const numElemsInCol = params.hiddenDim / FINALIZE_ELEM_PER_THREAD;
  bool const useScale = params.expertWeightsPtr != nullptr;

  __shared__ ScalePackedType scaleArrSmem[MaxTopK / TopKUnrollFactor];
  __shared__ IdxPackedType permutedIdxArrSmem[MaxTopK / TopKUnrollFactor];

  for (int kChunkIdx = threadIdx.x; kChunkIdx < params.topK / TopKUnrollFactor;
       kChunkIdx += blockDim.x) {
    int const expandedIdx = tokenIdx * params.topK + kChunkIdx * TopKUnrollFactor;
    auto permutedIdxPacked = reinterpret_cast<IdxPackedType const*>(
        params.expandedIdxToPermutedIdx)[expandedIdx / TopKUnrollFactor];
    auto scalePacked = useScale ? reinterpret_cast<ScalePackedType const*>(
                                      params.expertWeightsPtr)[expandedIdx / TopKUnrollFactor]
                                : ScalePackedType{TypeExpW(1.f)};

    scaleArrSmem[kChunkIdx] = scalePacked;
    permutedIdxArrSmem[kChunkIdx] = permutedIdxPacked;
  }

  auto const offset = tokenIdx * params.hiddenDim;
  Type* outputPtr = params.outPtr + offset;
  auto* outElemPtr = reinterpret_cast<OutputElem*>(outputPtr);
  auto const* inElemPtr = reinterpret_cast<InputElem const*>(params.inPtr);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif
  __syncthreads();

  for (int elemIndex = startOffset; elemIndex < numElemsInCol; elemIndex += stride) {
    ComputeElem threadOutput;
    threadOutput.fill(0);
    for (int kChunkIdx = 0; kChunkIdx < params.topK / TopKUnrollFactor; kChunkIdx++) {
      auto permutedIdxArr = *reinterpret_cast<IdxArrayType const*>(&permutedIdxArrSmem[kChunkIdx]);
      InputElem inputElemArr[TopKUnrollFactor];
#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        auto const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }

        auto const* inputPermutedPtr = inElemPtr + permutedIdx * numElemsInPaddedCol;

        float4 input =
            vectorizedLoadPtx(reinterpret_cast<float4 const*>(&inputPermutedPtr[elemIndex]));
        inputElemArr[ki] = *reinterpret_cast<InputElem const*>(&input);
      }
      auto scaleArr = *reinterpret_cast<ScaleArrayType const*>(&scaleArrSmem[kChunkIdx]);
      auto const scaleFloatArr =
          arrayConvert<ScaleArrayType, cutlass::Array<float, TopKUnrollFactor>>(scaleArr);

#pragma unroll
      for (int ki = 0; ki < TopKUnrollFactor; ++ki) {
        auto const permutedIdx = permutedIdxArr[ki];
        if (permutedIdx == -1) {
          continue;
        }
        auto scale = useScale ? scaleFloatArr[ki] : 1.0f;
        ComputeElem expertResult = arrayConvert<InputElem, ComputeElem>(inputElemArr[ki]);
        threadOutput = threadOutput + scale * expertResult;
      }
    }
    OutputElem outputElem = arrayConvert<ComputeElem, OutputElem>(threadOutput);
    outElemPtr[elemIndex] = outputElem;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeDeepSeekKernel(KernelParams params) {
  using Type = typename KernelParams::Type;
  using BlockReduce = cub::BlockReduce<float, 128>;

  __shared__ float s_scaleOut;
  __shared__ typename BlockReduce::TempStorage temp_storage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // wait on primary kernel when using PDL
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y) {
    // Loop over hidden dim
    for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
         hiddenIdx += blockDim.x * gridDim.x) {
      // Accumulate chunk of token into registers
      float acc = 0.0f;

      for (int k = 0; k < params.topK; k++) {
        int const expandedIdx = tokenIdx * params.topK + k;
        int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
        if (permutedIdx == -1) {
          continue;
        }
        int const totalNumPaddedTokens = params.totalNumPaddedTokens[0];
        int const scaleIdx = permutedIdx + totalNumPaddedTokens * (hiddenIdx / 128);
        float const blockScale = params.inDqSfsPtr ? params.inDqSfsPtr[scaleIdx] : 1;

        float const expertProb = (float)params.expertWeightsPtr[tokenIdx * params.topK + k];

        float const scale = expertProb * blockScale;
        acc += scale *
               static_cast<float>(params.inPtr[permutedIdx * params.hiddenDimPadded + hiddenIdx]);
      }

      // The largest (finite) value that can be represented using E4m3.
      float constexpr E4m3MaxVal{448.f};

      // Compute the absolute max
      float aMax = BlockReduce(temp_storage).Reduce(fabsf(acc), cuda::maximum<>{});

      if (threadIdx.x == 0) {
        if (params.outDqSfsPtr) {
          s_scaleOut = aMax / E4m3MaxVal;
          int const scaleOut_idx = tokenIdx + hiddenIdx / 128 * params.numTokens;
          params.outDqSfsPtr[scaleOut_idx] = aMax / E4m3MaxVal;
        } else {
          s_scaleOut = 1.0f;
        }
      }
      __syncthreads();
      float const scaleOut = s_scaleOut;
      __syncthreads();
      params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] = (Type)(acc / scaleOut);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

bool finalizerDiagnosticEnabled() {
  auto const* value = std::getenv("FLASHINFER_DIAGNOSTIC_MOE_FINALIZER");
  return value != nullptr && std::string(value) == "1";
}

bool shouldRunFinalizerDiagnostic(Data const& data) {
  return finalizerDiagnosticEnabled() && data.numTokens >= 32768;
}

size_t dtypeSize(tg::Dtype dtype) {
  switch (dtype) {
    case tg::Dtype::Fp16:
    case tg::Dtype::Bfloat16:
      return 2;
    case tg::Dtype::E4m3:
      return 1;
    case tg::Dtype::Fp32:
      return 4;
    default:
      return 0;
  }
}

uint64_t checkedProduct(std::initializer_list<uint64_t> factors, char const* name) {
  uint64_t result = 1;
  for (auto const factor : factors) {
    FLASHINFER_CHECK(factor == 0 || result <= std::numeric_limits<uint64_t>::max() / factor,
                     "[FI_FINALIZER_DIAGNOSTIC] byte-size overflow for ", name);
    result *= factor;
  }
  return result;
}

struct PointerRange {
  bool available;
  CUdeviceptr base;
  size_t size;
};

PointerRange logPointer(char const* name, void const* ptr, uint64_t requiredBytes,
                        uint64_t invocation) {
  if (ptr == nullptr) {
    std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation << " pointer=" << name
              << " address=null required_bytes=" << requiredBytes << std::endl;
    return {false, 0, 0};
  }

  cudaPointerAttributes attributes{};
  auto const attributeStatus = cudaPointerGetAttributes(&attributes, ptr);
  if (attributeStatus != cudaSuccess) {
    std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation << " pointer=" << name
              << " address=" << ptr << " required_bytes=" << requiredBytes
              << " alignment16=" << (reinterpret_cast<uintptr_t>(ptr) % 16)
              << " cuda_pointer_status=" << cudaGetErrorString(attributeStatus) << std::endl;
    cudaGetLastError();
  } else {
    std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation << " pointer=" << name
              << " address=" << ptr << " required_bytes=" << requiredBytes
              << " alignment16=" << (reinterpret_cast<uintptr_t>(ptr) % 16)
              << " memory_type=" << static_cast<int>(attributes.type)
              << " device=" << attributes.device << std::endl;
  }

  CUdeviceptr base = 0;
  size_t size = 0;
  auto const rangeStatus =
      cuMemGetAddressRange(&base, &size, reinterpret_cast<CUdeviceptr>(ptr));
  if (rangeStatus != CUDA_SUCCESS) {
    std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation << " pointer=" << name
              << " allocation_range=unavailable cu_status=" << static_cast<int>(rangeStatus)
              << std::endl;
    return {false, 0, 0};
  }

  auto const address = reinterpret_cast<uintptr_t>(ptr);
  auto const allocationEnd = static_cast<uint64_t>(base) + size;
  auto const requiredEnd = static_cast<uint64_t>(address) + requiredBytes;
  auto const contains = address >= static_cast<uintptr_t>(base) && requiredEnd >= address &&
                        requiredEnd <= allocationEnd;
  std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation << " pointer=" << name
            << " allocation_base="
            << reinterpret_cast<void*>(static_cast<uintptr_t>(base))
            << " allocation_bytes=" << size
            << " required_end=" << reinterpret_cast<void*>(static_cast<uintptr_t>(requiredEnd))
            << " contains_required=" << contains << std::endl;
  return {true, base, size};
}

bool containsRequired(PointerRange range, void const* ptr, uint64_t requiredBytes) {
  if (!range.available) {
    return true;
  }
  auto const address = reinterpret_cast<uintptr_t>(ptr);
  auto const requiredEnd = static_cast<uint64_t>(address) + requiredBytes;
  return address >= static_cast<uintptr_t>(range.base) && requiredEnd >= address &&
         requiredEnd <= static_cast<uint64_t>(range.base) + range.size;
}

void runFinalizerDiagnostic(Data const& data, void* stream, uint64_t invocation) {
  FLASHINFER_CHECK(data.numTokens >= 0, "[FI_FINALIZER_DIAGNOSTIC] negative numTokens");
  FLASHINFER_CHECK(data.topK > 0, "[FI_FINALIZER_DIAGNOSTIC] non-positive topK");
  FLASHINFER_CHECK(data.hiddenDim > 0, "[FI_FINALIZER_DIAGNOSTIC] non-positive hiddenDim");
  FLASHINFER_CHECK(data.hiddenDimPadded >= data.hiddenDim,
                   "[FI_FINALIZER_DIAGNOSTIC] hiddenDimPadded is smaller than hiddenDim");
  FLASHINFER_CHECK(data.inPtr != nullptr, "[FI_FINALIZER_DIAGNOSTIC] inPtr is null");
  FLASHINFER_CHECK(data.outPtr != nullptr, "[FI_FINALIZER_DIAGNOSTIC] outPtr is null");
  FLASHINFER_CHECK(data.totalNumPaddedTokens != nullptr,
                   "[FI_FINALIZER_DIAGNOSTIC] totalNumPaddedTokens is null");
  FLASHINFER_CHECK(data.expandedIdxToPermutedIdx != nullptr,
                   "[FI_FINALIZER_DIAGNOSTIC] expandedIdxToPermutedIdx is null");

  int device = -1;
  CHECK_CUDA_ERROR(cudaGetDevice(&device));
  std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation
            << " phase=pre_sync device=" << device << " stream=" << stream
            << " num_tokens=" << data.numTokens << " num_experts=" << data.numExperts
            << " top_k=" << data.topK << " hidden_dim=" << data.hiddenDim
            << " hidden_dim_padded=" << data.hiddenDimPadded
            << " use_pdl=" << data.mUsePdl << std::endl;

  auto const preSyncStatus = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  if (preSyncStatus != cudaSuccess) {
    std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation
              << " phase=pre_sync status=" << cudaGetErrorString(preSyncStatus) << std::endl;
  }
  FLASHINFER_CHECK(preSyncStatus == cudaSuccess,
                   "[FI_FINALIZER_DIAGNOSTIC] CUDA failed before finalizer launch");

  int32_t totalNumPaddedTokens = -1;
  CHECK_CUDA_ERROR(cudaMemcpy(&totalNumPaddedTokens, data.totalNumPaddedTokens, sizeof(int32_t),
                              cudaMemcpyDeviceToHost));
  FLASHINFER_CHECK(totalNumPaddedTokens >= 0,
                   "[FI_FINALIZER_DIAGNOSTIC] negative totalNumPaddedTokens");

  auto const mapEntries =
      checkedProduct({static_cast<uint64_t>(data.numTokens), static_cast<uint64_t>(data.topK)},
                     "permutation map entries");
  FLASHINFER_CHECK(mapEntries <= std::numeric_limits<size_t>::max() / sizeof(int32_t),
                   "[FI_FINALIZER_DIAGNOSTIC] permutation map is too large for host inspection");
  std::vector<int32_t> permutationMap(static_cast<size_t>(mapEntries));
  CHECK_CUDA_ERROR(cudaMemcpy(permutationMap.data(), data.expandedIdxToPermutedIdx,
                              permutationMap.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

  int32_t minimum = std::numeric_limits<int32_t>::max();
  int32_t maximum = std::numeric_limits<int32_t>::min();
  uint64_t sentinelCount = 0;
  uint64_t invalidNegativeCount = 0;
  uint64_t invalidUpperCount = 0;
  size_t firstInvalidIndex = permutationMap.size();
  int32_t firstInvalidValue = 0;
  for (size_t index = 0; index < permutationMap.size(); ++index) {
    auto const value = permutationMap[index];
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
    if (value == -1) {
      ++sentinelCount;
    } else if (value < 0) {
      ++invalidNegativeCount;
      if (firstInvalidIndex == permutationMap.size()) {
        firstInvalidIndex = index;
        firstInvalidValue = value;
      }
    } else if (value >= totalNumPaddedTokens) {
      ++invalidUpperCount;
      if (firstInvalidIndex == permutationMap.size()) {
        firstInvalidIndex = index;
        firstInvalidValue = value;
      }
    }
  }
  if (permutationMap.empty()) {
    minimum = 0;
    maximum = 0;
  }

  auto const outputElementSize = dtypeSize(data.mDtypeElt);
  auto const expertWeightElementSize = dtypeSize(data.mDtypeExpW);
  FLASHINFER_CHECK(outputElementSize != 0,
                   "[FI_FINALIZER_DIAGNOSTIC] unsupported output dtype");
  FLASHINFER_CHECK(data.expertWeightsPtr == nullptr || expertWeightElementSize != 0,
                   "[FI_FINALIZER_DIAGNOSTIC] unsupported expert-weight dtype");

  auto const inputBytes =
      checkedProduct({static_cast<uint64_t>(totalNumPaddedTokens),
                      static_cast<uint64_t>(data.hiddenDimPadded), outputElementSize},
                     "finalizer input");
  auto const outputBytes =
      checkedProduct({static_cast<uint64_t>(data.numTokens),
                      static_cast<uint64_t>(data.hiddenDim), outputElementSize},
                     "finalizer output");
  auto const mapBytes = checkedProduct({mapEntries, sizeof(int32_t)}, "permutation map");
  auto const expertWeightBytes =
      data.expertWeightsPtr == nullptr
          ? 0
          : checkedProduct({mapEntries, expertWeightElementSize}, "expert weights");

  std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation
            << " phase=metadata total_num_padded_tokens=" << totalNumPaddedTokens
            << " map_entries=" << mapEntries << " map_min=" << minimum
            << " map_max=" << maximum << " sentinel_count=" << sentinelCount
            << " invalid_negative_count=" << invalidNegativeCount
            << " invalid_upper_count=" << invalidUpperCount;
  if (firstInvalidIndex != permutationMap.size()) {
    std::cerr << " first_invalid_index=" << firstInvalidIndex
              << " first_invalid_value=" << firstInvalidValue;
  }
  std::cerr << std::endl;

  auto const inputRange = logPointer("in_ptr", data.inPtr, inputBytes, invocation);
  auto const outputRange = logPointer("out_ptr", data.outPtr, outputBytes, invocation);
  auto const mapRange =
      logPointer("expanded_idx_to_permuted_idx", data.expandedIdxToPermutedIdx, mapBytes,
                 invocation);
  logPointer("total_num_padded_tokens", data.totalNumPaddedTokens, sizeof(int32_t), invocation);
  PointerRange expertWeightRange{false, 0, 0};
  if (data.expertWeightsPtr != nullptr) {
    expertWeightRange =
        logPointer("expert_weights", data.expertWeightsPtr, expertWeightBytes, invocation);
  }

  FLASHINFER_CHECK(containsRequired(inputRange, data.inPtr, inputBytes),
                   "[FI_FINALIZER_DIAGNOSTIC] input allocation does not contain required bytes");
  FLASHINFER_CHECK(containsRequired(outputRange, data.outPtr, outputBytes),
                   "[FI_FINALIZER_DIAGNOSTIC] output allocation does not contain required bytes");
  FLASHINFER_CHECK(
      containsRequired(mapRange, data.expandedIdxToPermutedIdx, mapBytes),
      "[FI_FINALIZER_DIAGNOSTIC] permutation-map allocation does not contain required bytes");
  FLASHINFER_CHECK(data.expertWeightsPtr == nullptr ||
                       containsRequired(expertWeightRange, data.expertWeightsPtr, expertWeightBytes),
                   "[FI_FINALIZER_DIAGNOSTIC] expert-weight allocation does not contain required "
                   "bytes");
  FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(data.inPtr) % 16 == 0,
                   "[FI_FINALIZER_DIAGNOSTIC] finalizer input is not 16-byte aligned");
  FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(data.outPtr) % 16 == 0,
                   "[FI_FINALIZER_DIAGNOSTIC] finalizer output is not 16-byte aligned");
  FLASHINFER_CHECK(reinterpret_cast<uintptr_t>(data.expandedIdxToPermutedIdx) % 16 == 0,
                   "[FI_FINALIZER_DIAGNOSTIC] permutation map is not 16-byte aligned");
  FLASHINFER_CHECK(invalidNegativeCount == 0 && invalidUpperCount == 0,
                   "[FI_FINALIZER_DIAGNOSTIC] permutation map contains out-of-range entries");

  std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation
            << " phase=validated status=ok" << std::endl;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  static std::atomic<uint64_t> diagnosticInvocation{0};
  auto const diagnosticEnabled = shouldRunFinalizerDiagnostic(data);
  auto const invocation =
      diagnosticEnabled ? diagnosticInvocation.fetch_add(1, std::memory_order_relaxed) : 0;
  if (diagnosticEnabled) {
    runFinalizerDiagnostic(data, stream, invocation);
  }

  if (data.mUseDeepSeekFp8) {
    int const numThreads = 128;
    int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
    // Capped at rather arbitrary 8192 to avoid gridDim exceeding 65535 specified by CUDA.
    int const numBlocksY = std::min(8192, data.numTokens);
    dim3 numBlocks(numBlocksX, numBlocksY);

    LAUNCH_TOPK_EXPW(data, finalizeDeepSeekKernel, numBlocks, numThreads, 0, stream);
  } else {
    int const numThreads = 256;
    int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
    // Capped at rather arbitrary 8192 to avoid gridDim exceeding 65535 specified by CUDA.
    int const numBlocksY = std::min(8192, data.numTokens);

    if (numBlocksX * numBlocksY < 1184) {
      // The number 1184 comes from 148 * 8, where 148 is the number of SMs (Streaming
      // Multiprocessors) in the Blackwell architecture, and the value 8 means that each Streaming
      // Multiprocessor (SM) can hold up to 8 blocks for this kernel. This limitation is intended to
      // ensure that when the number of waves is greater than 1, we choose to use the kernel with
      // vectorized loading.
      dim3 numBlocks(numBlocksX, numBlocksY);
      LAUNCH_TOPK_EXPW(data, finalizeKernel, numBlocks, numThreads, 0, stream);
    } else {
      FLASHINFER_CHECK(
          data.topK <= MaxTopK,
          "Finalize kernel with vectorized loading is not supported for this TopK value: %d",
          data.topK);
      LAUNCH_TOPK_EXPW(data, finalizeKernelVecLoad, /*numBlocks=*/data.numTokens,
                       /*numThreads=*/FINALIZE_THREADS_PER_BLOCK, 0, stream);
    }
  }

  if (diagnosticEnabled) {
    auto const postSyncStatus = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    if (postSyncStatus != cudaSuccess) {
      std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation
                << " phase=post_sync status=" << cudaGetErrorString(postSyncStatus) << std::endl;
    }
    FLASHINFER_CHECK(postSyncStatus == cudaSuccess,
                     "[FI_FINALIZER_DIAGNOSTIC] CUDA failed in finalizer launch");
    std::cerr << "[FI_FINALIZER_DIAGNOSTIC] invocation=" << invocation
              << " phase=post_sync status=ok" << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace finalize

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace moe::dev
