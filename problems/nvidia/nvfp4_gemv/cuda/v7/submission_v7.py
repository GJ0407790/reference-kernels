# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

gemv_cuda_src = """
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \\
    {                                                                           \\
        const cudaError_t error = call;                                         \\
        if (error != cudaSuccess)                                               \\
        {                                                                       \\
            printf("Error: %s:%d, ", __FILE__, __LINE__);                       \\
            printf("code:%d, reason: %s\\n", error, cudaGetErrorString(error)); \\
            exit(1);                                                            \\
        }                                                                       \\
    }

using Reg128 = uint4;
using Reg32 = uint32_t;
using Reg16 = uint16_t;
using f16 = __half;
using uint8 = uint8_t;
using fp8_e4m3 = __nv_fp8_e4m3;

// Taken from https://github.com/NVIDIA/TransformerEngine/blob/9c3cb1fbb62970304d1d84a14084c6d25e1983ac/transformer_engine/common/util/ptx.cuh
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, const uint32_t count) 
{
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mbar_ptr), "r"(count) : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
__device__ __forceinline__ void mbarrier_invalid(uint64_t* mbar) 
{
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(mbar_ptr) : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) 
{
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(mbar_ptr) : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, const uint32_t tx_count) 
{
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(mbar_ptr), "r"(tx_count)
               : "memory");
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx
__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* mbar, const uint32_t tx_count) 
{
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.expect_tx.shared.b64 [%0], %1;" ::"r"(mbar_ptr), "r"(tx_count)
               : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() 
{
  asm volatile("fence.mbarrier_init.release.cluster;");
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint32_t mbar_ptr, const uint32_t parity) 
{
  uint32_t waitComplete;
  asm volatile(
      "{\\n\\t .reg .pred P_OUT; \\n\\t"
      "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2; \\n\\t"
      "selp.b32 %0, 1, 0, P_OUT; \\n"
      "}"
      : "=r"(waitComplete)
      : "r"(mbar_ptr), "r"(parity)
      : "memory");
  return static_cast<bool>(waitComplete);
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, const uint32_t parity) 
{
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  while (!mbarrier_try_wait_parity(mbar_ptr, parity)) {}
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// global -> shared::cluster
template<const int policy>
__device__ __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared(
    uint64_t* dst_shmem, 
    const uint64_t* tensor_map_ptr, 
    const uint32_t offset_x,
    const uint32_t offset_y, 
    uint64_t* mbar
)
{
  uint32_t dst_shmem_ptr = __cvta_generic_to_shared(dst_shmem);
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  // triggers async copy, i.e. the thread continues until wait() on mbarrier
  // barrier condition:
  // - leader must arrive (i.e. 1 thread as set above)
  // - TMA hardware substracts bytes from expect_tx counter, must reach zero
  constexpr uint64_t cache_policy_evict_normal = 0x0ULL; // Normal caching
  constexpr uint64_t cache_policy_evict_first = 0x1ULL;  // Evict first
  constexpr uint64_t cache_policy_evict_last = 0x2ULL;   // Evict last (keep in cache)
  constexpr uint64_t cache_policy_no_allocate = 0x3ULL;   // No allocate

  if constexpr (policy == 0) // normal
  {
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile"
      ".mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst_shmem_ptr),
      "l"(tensor_map_ptr), "r"(offset_x), "r"(offset_y), "r"(mbar_ptr), "l"(cache_policy_evict_normal)
      : "memory");
  } 
  else if constexpr (policy == 1) // evict first
  {
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile"
      ".mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst_shmem_ptr),
      "l"(tensor_map_ptr), "r"(offset_x), "r"(offset_y), "r"(mbar_ptr), "l"(cache_policy_evict_first)
      : "memory");
  } 
  else if constexpr (policy == 2) // evict last
  {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst_shmem_ptr),
        "l"(tensor_map_ptr), "r"(offset_x), "r"(offset_y), "r"(mbar_ptr), "l"(cache_policy_evict_last)
        : "memory");
  }
  else if constexpr (policy == 3) // no allocate
  {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile"
        ".mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst_shmem_ptr),
        "l"(tensor_map_ptr), "r"(offset_x), "r"(offset_y), "r"(mbar_ptr), "l"(cache_policy_no_allocate)
        : "memory");
  }
  else
  {
    // invalid policy
    static_assert(policy >= 0 && policy <=3, "Invalid cache policy for cp.async bulk tensor");
  }
}

void create_2D_tensor_map(
  CUtensorMap& tensorMap, 
  void* data_ptr,
  const uint64_t global_height, 
  const uint64_t global_width, 
  const uint32_t smem_height,
  const uint32_t smem_width, 
  const uint64_t stride_bytes) 
{
  // Get a function pointer to the cuTensorMapEncodeTiled driver API
  // Note: PFN_cuTensorMapEncodeTiled is not defined in cuda13
  static PFN_cuTensorMapEncodeTiled_v12000 cuDriverTensorMapEncodeTiled = []() {
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
  }();

  // rank is the number of dimensions of the array
  constexpr uint32_t rank = 2;

  // Dimension for the packed data types must reflect the number of individual U# values.
  uint64_t size[rank] = {global_width, global_height};

  // The stride is the number of bytes to traverse from the first element of one row to the next
  uint64_t stride[rank - 1] = {stride_bytes};

  // The boxSize is the size of the shared memory buffer that is used as the
  // source/destination of a TMA transfer
  uint32_t boxSize[rank] = {smem_width, smem_height};

  // The distance between elements in units of sizeof(element)
  uint32_t elemStride[rank] = {1, 1};

  // Create the tensor descriptor.
  cuDriverTensorMapEncodeTiled(
      &tensorMap,  // CUtensorMap *tensorMap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
      rank,        // cuuint32_t tensorRank,
      data_ptr,     // void *globalAddress,
      size,        // const cuuint64_t *globalDim,
      stride,      // const cuuint64_t *globalStrides,
      boxSize,     // const cuuint32_t *boxDim,
      elemStride,  // const cuuint32_t *elementStrides,
      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,

      // Swizzling can be used to avoid shared memory bank conflicts.
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,

      // L2 Promotion can be used to widen the effect of a cache-policy to a wider
      // set of L2 cache lines.
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,

      // Any element that is outside of bounds will be set to zero by the TMA transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// taken from https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/gemv_blockscaled.h#L564
__device__ f16 blockscaled_multiply_add(
  const Reg32& a0, const Reg32& a1, const Reg32& a2, const Reg32& a3,
  const Reg32& b0, const Reg32& b1, const Reg32& b2, const Reg32& b3,
  const Reg16& sfa,
  const Reg16& sfb
)
{
  f16 res;
  Reg16* res_u16_ptr = reinterpret_cast<Reg16*>(&res); 

  asm volatile( \\
    "{\\n" \\
    // declare registers for A / B tensors
    ".reg .b8 byte0_0, byte0_1, byte0_2, byte0_3;\\n" \\
    ".reg .b8 byte0_4, byte0_5, byte0_6, byte0_7;\\n" \\
    ".reg .b8 byte1_0, byte1_1, byte1_2, byte1_3;\\n" \\
    ".reg .b8 byte1_4, byte1_5, byte1_6, byte1_7;\\n" \\
    ".reg .b8 byte2_0, byte2_1, byte2_2, byte2_3;\\n" \\
    ".reg .b8 byte2_4, byte2_5, byte2_6, byte2_7;\\n" \\
    ".reg .b8 byte3_0, byte3_1, byte3_2, byte3_3;\\n" \\
    ".reg .b8 byte3_4, byte3_5, byte3_6, byte3_7;\\n" \\

    // declare registers for accumulators
    ".reg .f16x2 accum_0_0, accum_0_1, accum_0_2, accum_0_3;\\n" \\
    ".reg .f16x2 accum_1_0, accum_1_1, accum_1_2, accum_1_3;\\n" \\
    ".reg .f16x2 accum_2_0, accum_2_1, accum_2_2, accum_2_3;\\n" \\
    ".reg .f16x2 accum_3_0, accum_3_1, accum_3_2, accum_3_3;\\n" \\

    // declare registers for scaling factors
    ".reg .f16x2 sfa_f16x2;\\n" \\
    ".reg .f16x2 sfb_f16x2;\\n" \\
    ".reg .f16x2 sf_f16x2;\\n" \\

    // declare registers for conversion
    ".reg .f16x2 cvt_0_0, cvt_0_1, cvt_0_2, cvt_0_3;\\n" \\
    ".reg .f16x2 cvt_0_4, cvt_0_5, cvt_0_6, cvt_0_7;\\n" \\
    ".reg .f16x2 cvt_1_0, cvt_1_1, cvt_1_2, cvt_1_3;\\n" \\
    ".reg .f16x2 cvt_1_4, cvt_1_5, cvt_1_6, cvt_1_7;\\n" \\
    ".reg .f16x2 cvt_2_0, cvt_2_1, cvt_2_2, cvt_2_3;\\n" \\
    ".reg .f16x2 cvt_2_4, cvt_2_5, cvt_2_6, cvt_2_7;\\n" \\
    ".reg .f16x2 cvt_3_0, cvt_3_1, cvt_3_2, cvt_3_3;\\n" \\
    ".reg .f16x2 cvt_3_4, cvt_3_5, cvt_3_6, cvt_3_7;\\n" \\
    ".reg .f16 result_f16, lane0, lane1;\\n" \\
    ".reg .f16x2 mul_f16x2_0, mul_f16x2_1;\\n" \\

    // convert scaling factors from fp8 to f16x2
    "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %1;\\n" \\
    "cvt.rn.f16x2.e4m3x2 sfb_f16x2, %2;\\n" \\
    
    // clear accumulators
    "mov.b32 accum_0_0, 0;\\n" \\
    "mov.b32 accum_0_1, 0;\\n" \\
    "mov.b32 accum_0_2, 0;\\n" \\
    "mov.b32 accum_0_3, 0;\\n" \\
    "mov.b32 accum_1_0, 0;\\n" \\
    "mov.b32 accum_1_1, 0;\\n" \\
    "mov.b32 accum_1_2, 0;\\n" \\
    "mov.b32 accum_1_3, 0;\\n" \\
    "mov.b32 accum_2_0, 0;\\n" \\
    "mov.b32 accum_2_1, 0;\\n" \\
    "mov.b32 accum_2_2, 0;\\n" \\
    "mov.b32 accum_2_3, 0;\\n" \\
    "mov.b32 accum_3_0, 0;\\n" \\
    "mov.b32 accum_3_1, 0;\\n" \\
    "mov.b32 accum_3_2, 0;\\n" \\
    "mov.b32 accum_3_3, 0;\\n" \\

    // multiply, unpacking and permuting scale factors
    "mul.rn.f16x2 sf_f16x2, sfa_f16x2, sfb_f16x2;\\n" \\
    "mov.b32 {lane0, lane1}, sf_f16x2;\\n" \\
    "mov.b32 mul_f16x2_0, {lane0, lane0};\\n" \\
    "mov.b32 mul_f16x2_1, {lane1, lane1};\\n" \\

    // unpacking A and B tensors
    "mov.b32 {byte0_0, byte0_1, byte0_2, byte0_3}, %3;\\n" \\
    "mov.b32 {byte0_4, byte0_5, byte0_6, byte0_7}, %4;\\n" \\
    "mov.b32 {byte1_0, byte1_1, byte1_2, byte1_3}, %5;\\n" \\
    "mov.b32 {byte1_4, byte1_5, byte1_6, byte1_7}, %6;\\n" \\
    "mov.b32 {byte2_0, byte2_1, byte2_2, byte2_3}, %7;\\n" \\
    "mov.b32 {byte2_4, byte2_5, byte2_6, byte2_7}, %8;\\n" \\
    "mov.b32 {byte3_0, byte3_1, byte3_2, byte3_3}, %9;\\n" \\
    "mov.b32 {byte3_4, byte3_5, byte3_6, byte3_7}, %10;\\n" \\

    // convert A and B tensors from fp4 to f16x2

    // A[0 - 7] and B[0 - 7]
    "cvt.rn.f16x2.e2m1x2 cvt_0_0, byte0_0;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_0_1, byte0_1;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_0_2, byte0_2;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_0_3, byte0_3;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_0_4, byte0_4;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_0_5, byte0_5;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_0_6, byte0_6;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_0_7, byte0_7;\\n" \\

    // A[8 - 15] and B[8 - 15]
    "cvt.rn.f16x2.e2m1x2 cvt_1_0, byte1_0;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_1_1, byte1_1;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_1_2, byte1_2;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_1_3, byte1_3;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_1_4, byte1_4;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_1_5, byte1_5;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_1_6, byte1_6;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_1_7, byte1_7;\\n" \\

    // A[16 - 23] and B[16 - 23]
    "cvt.rn.f16x2.e2m1x2 cvt_2_0, byte2_0;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_2_1, byte2_1;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_2_2, byte2_2;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_2_3, byte2_3;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_2_4, byte2_4;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_2_5, byte2_5;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_2_6, byte2_6;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_2_7, byte2_7;\\n" \\

    // A[24 - 31] and B[24 - 31]
    "cvt.rn.f16x2.e2m1x2 cvt_3_0, byte3_0;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_3_1, byte3_1;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_3_2, byte3_2;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_3_3, byte3_3;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_3_4, byte3_4;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_3_5, byte3_5;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_3_6, byte3_6;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_3_7, byte3_7;\\n" \\

    // fma for A[0 - 7] and B[0 - 7]
    "fma.rn.f16x2 accum_0_0, cvt_0_0, cvt_0_4, accum_0_0;\\n" \\
    "fma.rn.f16x2 accum_0_1, cvt_0_1, cvt_0_5, accum_0_1;\\n" \\
    "fma.rn.f16x2 accum_0_2, cvt_0_2, cvt_0_6, accum_0_2;\\n" \\
    "fma.rn.f16x2 accum_0_3, cvt_0_3, cvt_0_7, accum_0_3;\\n" \\

    // fma for A[8 - 15] and B[8 - 15]
    "fma.rn.f16x2 accum_1_0, cvt_1_0, cvt_1_4, accum_1_0;\\n" \\
    "fma.rn.f16x2 accum_1_1, cvt_1_1, cvt_1_5, accum_1_1;\\n" \\
    "fma.rn.f16x2 accum_1_2, cvt_1_2, cvt_1_6, accum_1_2;\\n" \\
    "fma.rn.f16x2 accum_1_3, cvt_1_3, cvt_1_7, accum_1_3;\\n" \\

    // fma for A[16 - 23] and B[16 - 23]
    "fma.rn.f16x2 accum_2_0, cvt_2_0, cvt_2_4, accum_2_0;\\n" \\
    "fma.rn.f16x2 accum_2_1, cvt_2_1, cvt_2_5, accum_2_1;\\n" \\
    "fma.rn.f16x2 accum_2_2, cvt_2_2, cvt_2_6, accum_2_2;\\n" \\
    "fma.rn.f16x2 accum_2_3, cvt_2_3, cvt_2_7, accum_2_3;\\n" \\

    // fma for A[24 - 31] and B[24 - 31]
    "fma.rn.f16x2 accum_3_0, cvt_3_0, cvt_3_4, accum_3_0;\\n" \\
    "fma.rn.f16x2 accum_3_1, cvt_3_1, cvt_3_5, accum_3_1;\\n" \\
    "fma.rn.f16x2 accum_3_2, cvt_3_2, cvt_3_6, accum_3_2;\\n" \\
    "fma.rn.f16x2 accum_3_3, cvt_3_3, cvt_3_7, accum_3_3;\\n" \\

    // tree reduction for accumulators
    "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_1;\\n" \\
    "add.rn.f16x2 accum_0_2, accum_0_2, accum_0_3;\\n" \\
    "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_1;\\n" \\
    "add.rn.f16x2 accum_1_2, accum_1_2, accum_1_3;\\n" \\
    "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_1;\\n" \\
    "add.rn.f16x2 accum_2_2, accum_2_2, accum_2_3;\\n" \\
    "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_1;\\n" \\
    "add.rn.f16x2 accum_3_2, accum_3_2, accum_3_3;\\n" \\

    "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_2;\\n" \\
    "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_2;\\n" \\
    "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_2;\\n" \\
    "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_2;\\n" \\

    "add.rn.f16x2 accum_0_0, accum_0_0, accum_1_0;\\n" \\
    "add.rn.f16x2 accum_2_0, accum_2_0, accum_3_0;\\n" \\

    // apply scaling factors and final reduction
    "mul.rn.f16x2 accum_0_0, mul_f16x2_0, accum_0_0;\\n" \\
    "mul.rn.f16x2 accum_2_0, mul_f16x2_1, accum_2_0;\\n" \\

    "add.rn.f16x2 accum_0_0, accum_0_0, accum_2_0;\\n" \\

    "mov.b32 {lane0, lane1}, accum_0_0;\\n" \\
    "add.rn.f16 result_f16, lane0, lane1;\\n" \\
    "mov.b16 %0, result_f16;\\n" \\

    "}\\n"
    : "=h"(res_u16_ptr[0])  // 0
    : "h"(sfa), "h"(sfb),   // 1, 2
      "r"(a0), "r"(b0),     // 3, 4
      "r"(a1), "r"(b1),     // 5, 6
      "r"(a2), "r"(b2),     // 7, 8
      "r"(a3), "r"(b3)      // 9, 10
    : "memory"
  );

  return res;
}

template<
  const int BK_PER_ITER,
  const int BK_ITER,
  const int BM_PER_ITER,
  const int BM_ITER,
  const int SFK,
  const int TK,
  const int STAGES,
  const int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void gemv_kernel(
  f16* __restrict__ c,
  const int B, const int M, const int K, const int N,
  const __grid_constant__ CUtensorMap tensor_map_a,
  const __grid_constant__ CUtensorMap tensor_map_b,
  const __grid_constant__ CUtensorMap tensor_map_sfa,
  const __grid_constant__ CUtensorMap tensor_map_sfb)
{
  constexpr int BM = BM_PER_ITER * BM_ITER;

  constexpr int TMA_XACT_DATA_A_BYTES = BM_PER_ITER * BK_PER_ITER/2; // for a
  constexpr int TMA_XACT_DATA_B_BYTES = BK_PER_ITER/2; // for b
  constexpr int TMA_XACT_DATA_BYTES = TMA_XACT_DATA_A_BYTES + TMA_XACT_DATA_B_BYTES; // for a and b

  constexpr int TMA_XACT_SF_A_BYTES = BM_PER_ITER * SFK; // for sfas
  constexpr int TMA_XACT_SF_B_BYTES = SFK; // for sfbs
  constexpr int TMA_XACT_SF_BYTES = TMA_XACT_SF_A_BYTES + TMA_XACT_SF_B_BYTES; // for sfas and sfbs

  constexpr int TMA_ALL_A_BYTES = TMA_XACT_DATA_A_BYTES + TMA_XACT_SF_A_BYTES;
  constexpr int TMA_ALL_B_BYTES = TMA_XACT_DATA_B_BYTES + TMA_XACT_SF_B_BYTES;
  constexpr int TMA_ALL_BYTES = TMA_ALL_A_BYTES + TMA_ALL_B_BYTES;

  __shared__ extern char smem[];

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();

  uint8* as = reinterpret_cast<uint8*>(&smem[0]);
  uint8* bs = &as[STAGES * BM_PER_ITER * BK_PER_ITER/2];
  fp8_e4m3* sfas = reinterpret_cast<fp8_e4m3*>(&bs[BK_PER_ITER/2]);
  fp8_e4m3* sfbs = &sfas[STAGES * BM_PER_ITER * SFK];
  uint64_t* mbarrier = reinterpret_cast<uint64_t*>(&sfbs[SFK]);
  float* results = reinterpret_cast<float*>(&mbarrier[STAGES]);

  const int tid = threadIdx.x;
  const int a_row = blockIdx.z * M + blockIdx.x * BM;
  const int b_row = blockIdx.z * N;
  const int block_col = blockIdx.y * BK_ITER * BK_PER_ITER;
  int parity = 0;

  constexpr int THREADS_PER_K = BK_PER_ITER / TK;
  const int tcol = (tid % THREADS_PER_K) * TK;
  const int trow = tid / THREADS_PER_K;

  // accumulate on float to preserve precision
  float accum[BM_ITER] = {0}; 
  // b register is the same throughout BM_ITER
  Reg128 b_reg128; 
  Reg16 sfb_reg;

  // offset into the correct smem_position
  as += trow * BK_PER_ITER/2 + tcol / 2;
  bs += tcol / 2;
  sfas += trow * SFK + tcol / 16;
  sfbs += tcol / 16;

  // offset into correct c position
  c += a_row;

  // initialize mbarrier for tma
  if (tid == 0)
  {
    #pragma unroll
    for (int i = 0; i < STAGES; i++)
    {
      mbarrier_init(&mbarrier[i], BLOCK_SIZE);
    }

    fence_mbarrier_init_release_cluster();
  }

  __syncthreads();
  
  // prefetch first STAGES - 1 tiles
  // STAGES == BM_PER_ITER so will only prefetch on the same column
  for (int i = 0; i < STAGES; i++)
  {
    if (tid == 0)
    {
      cp_async_bulk_tensor_2d_global_to_shared<3 /*no_allocate*/>(
        reinterpret_cast<uint64_t*>(&as[i * BM_PER_ITER * BK_PER_ITER/2]),
        reinterpret_cast<const uint64_t*>(&tensor_map_a),
        block_col/2,
        a_row + i * BM_PER_ITER,
        &mbarrier[i]);

      cp_async_bulk_tensor_2d_global_to_shared<0 /*evict_normal*/>(
        reinterpret_cast<uint64_t*>(&sfas[i * BM_PER_ITER * SFK]),
        reinterpret_cast<const uint64_t*>(&tensor_map_sfa),
        block_col/16,
        a_row + i * BM_PER_ITER,
        &mbarrier[i]);
      
      // load b and sfb only in the first BM_ITER
      if (i == 0)
      {
        cp_async_bulk_tensor_2d_global_to_shared<2 /*evict_last*/>(
          reinterpret_cast<uint64_t*>(bs),
          reinterpret_cast<const uint64_t*>(&tensor_map_b),
          block_col/2,
          b_row,
          &mbarrier[i]);
        
        cp_async_bulk_tensor_2d_global_to_shared<2 /*evict_last*/>(
          reinterpret_cast<uint64_t*>(sfbs),
          reinterpret_cast<const uint64_t*>(&tensor_map_sfb),
          block_col/16,
          b_row,
          &mbarrier[i]);
        
        mbarrier_arrive_expect_tx(&mbarrier[i], TMA_ALL_BYTES);
      }
      else
      {
        mbarrier_arrive_expect_tx(&mbarrier[i], TMA_ALL_A_BYTES);
      }
    }
    else
    {
      // non first thread
      mbarrier_arrive(&mbarrier[i]);
    }
  }

  for (int k = 0; (k < BK_ITER) && (block_col + k * BK_PER_ITER < K); k++)
  {
    // start inner loop for a and sfa
    for (int i = 0; i < BM_ITER; i++)
    {
      // const int curr_iter = k * BM_ITER + i;
      // because STAGES == BM_PER_ITER, curr_iter % STAGES == i
      // hence we can use i directly
      mbarrier_wait_parity(&mbarrier[i], parity);

      // fma here
      // Load b once into the registers
      // then load into registers
      if (i == 0)
      {
        b_reg128 = reinterpret_cast<Reg128*>(bs)[0];
        sfb_reg = reinterpret_cast<Reg16*>(sfbs)[0];
      }
      
      Reg128 a_reg128 = reinterpret_cast<Reg128*>(&as[i * BM_PER_ITER * BK_PER_ITER/2])[0];
      Reg16 sfa_reg = reinterpret_cast<Reg16*>(&sfas[i * BM_PER_ITER * SFK])[0];

      Reg32* b_reg32_ptr = reinterpret_cast<Reg32*>(&b_reg128);
      Reg32* a_reg32_ptr = reinterpret_cast<Reg32*>(&a_reg128);

      f16 res = blockscaled_multiply_add(
                  a_reg32_ptr[0], a_reg32_ptr[1], a_reg32_ptr[2], a_reg32_ptr[3],
                  b_reg32_ptr[0], b_reg32_ptr[1], b_reg32_ptr[2], b_reg32_ptr[3],
                  sfa_reg,
                  sfb_reg
                );

      accum[i] +=  __half2float(res);

      __syncthreads();

      // prefetch next tile
      const int next_iter = k * BM_ITER + i + STAGES;

      if (next_iter < BK_ITER * BM_ITER)
      {
        const int next_i = next_iter % BM_ITER;
        const int next_k = next_iter / BM_ITER;
        const int next_col = block_col + next_k * BK_PER_ITER;

        if (next_col < K)
        {
          if (tid == 0)
          {
            cp_async_bulk_tensor_2d_global_to_shared<3 /*no_allocate*/>(
              reinterpret_cast<uint64_t*>(&as[next_i * BM_PER_ITER * BK_PER_ITER/2]),
              reinterpret_cast<const uint64_t*>(&tensor_map_a),
              next_col/2,
              a_row + next_i * BM_PER_ITER,
              &mbarrier[next_i]);

            if (next_i == 0)
            {
              cp_async_bulk_tensor_2d_global_to_shared<2 /*evict_last*/>(
                reinterpret_cast<uint64_t*>(bs),
                reinterpret_cast<const uint64_t*>(&tensor_map_b),
                next_col/2,
                b_row,
                &mbarrier[next_i]);
              
              mbarrier_arrive_expect_tx(&mbarrier[next_i], TMA_XACT_DATA_BYTES);
            }
            else
            {
              mbarrier_arrive_expect_tx(&mbarrier[next_i], TMA_XACT_DATA_A_BYTES);
            }
          }
          else
          {
            mbarrier_arrive(&mbarrier[next_i]);
          }
        }
      }
    } // end of BM_ITER

    // move to next sfb tile
    sfbs += BK_PER_ITER / 16; 
    sfas += BK_PER_ITER / 16;

    // toggle parity for mbarrier
    parity ^= 1;
  }

  // reduce within threads on the same k
  #pragma unroll
  for (int i = 0; i < BM_ITER; i++)
  {
    #pragma unroll
    for (int offset = 1; offset < THREADS_PER_K; offset <<= 1)
    {
      accum[i] += __shfl_xor_sync(0xFFFFFFFF, accum[i], offset);
    }
  }

  // store to shmem to do a cluster reduction
  if (tcol == 0)
  {
    #pragma unroll
    for (int i = 0; i < BM_ITER; i++)
    {
      results[trow + i * BM_PER_ITER] = accum[i];
    }
  }

  // wait for all blocks in cluster to finish storing to smem
  cluster.sync();

  float final_result;

  // only the first block in K dimension will do the reduction and store
  if (blockIdx.y == 0 && tid < BM)
  {
    final_result = results[tid];

    #pragma unroll
    for (int dst_block_rank = 1; dst_block_rank < gridDim.y; dst_block_rank++)
    {
      float* dst_results = cluster.map_shared_rank(results, dst_block_rank);
      final_result += dst_results[tid];
    }
  }

  // need to ensure first block is done reading
  cluster.sync();

  // write back result
  if (blockIdx.y == 0 && tid < BM)
  {
    c[tid] = __float2half(final_result);
  }

  // destroy mbarrier
  if (tid == 0)
  {
    #pragma unroll
    for (int i = 0; i < STAGES; i++)
    {
      mbarrier_invalid(&mbarrier[i]);
    }
  }
}

template<
  const int BK_PER_ITER, // Number of elements processed along K dimension in an iteration
  const int BK_ITER,     // Number of iterations along K dimension
  const int BM_PER_ITER, // Number of elements processed along M dimension in an iteration
  const int BM_ITER>     // Number of iterations along M dimension
void gemv(
  torch::Tensor a, 
  torch::Tensor b,
  torch::Tensor sfa,
  torch::Tensor sfb,
  torch::Tensor c)
{
  const int M = a.size(0);
  const int K = a.size(1) * 2; // a is in fp4, so each uint8 contains 2 elements
  const int B = a.size(2);
  const int N = b.size(0);

  // Each thread processes 32 elements along K dimension
  constexpr int TK = 32;

  // ensure we have enough scaling factors since we only load in 128 of them
  constexpr int BK = BK_PER_ITER * BK_ITER;

  // Number of scaling factors along K dimension loaded to shared memory
  // Want to hit 128B for cache line efficiency
  constexpr int SFK = BK / 16;

  constexpr int BM = BM_PER_ITER * BM_ITER;
  assert(M % BM == 0); // M must be divisible by BM

  constexpr int THREADS_PER_K = BK_PER_ITER / TK;
  constexpr int WM = 32 / THREADS_PER_K; // Number of rows per warp

  assert(BM_PER_ITER % WM == 0); // BM must be divisible by WM

  constexpr int NUM_WARPS = BM_PER_ITER / WM;
  constexpr int BLOCK_SIZE = NUM_WARPS * 32;
  
  const int NUM_BLOCKS_IN_K = (K - 1) / BK + 1;
  assert(NUM_BLOCKS_IN_K <= 8); // cluster limit to 8 blocks

  const dim3 grid(M / BM, NUM_BLOCKS_IN_K, B);

  // Tensormaps
  alignas(64) CUtensorMap tensor_map_a{};
  alignas(64) CUtensorMap tensor_map_b{};
  alignas(64) CUtensorMap tensor_map_sfa{};
  alignas(64) CUtensorMap tensor_map_sfb{};

  create_2D_tensor_map(
    tensor_map_a,
    a.data_ptr(),
    B*M,           // global height
    K/2,           // global width
    BM_PER_ITER,   // smem height
    BK_PER_ITER/2, // smem width
    K/2            // stride in bytes
  );

  create_2D_tensor_map(
    tensor_map_b,
    b.data_ptr(),
    N*B,           // global height
    K/2,           // global width
    1,             // smem height
    BK_PER_ITER/2, // smem width
    K/2            // stride in bytes
  );

  create_2D_tensor_map(
    tensor_map_sfa,
    sfa.data_ptr(),
    B*M,         // global height
    K/16,        // global width
    BM_PER_ITER, // smem height
    SFK,         // smem width
    K/16         // stride in bytes
  );

  create_2D_tensor_map(
    tensor_map_sfb,
    sfb.data_ptr(),
    N*B,        // global height
    K/16,       // global width
    1,          // smem height
    SFK,        // smem width
    K/16        // stride in bytes
  );

  constexpr int STAGES = BM_ITER; // let stages be enough for inner loop BM_ITER

  constexpr size_t SMEM_DATA = STAGES * (BM_PER_ITER * BK_PER_ITER/2) + BK_PER_ITER/2; // a + b
  constexpr size_t SMEM_SF = STAGES * (BM_PER_ITER * SFK) + SFK;                       // sfa + sfb
  constexpr size_t SMEM_BARRIER = STAGES * sizeof(uint64_t);                           // mbarriers
  constexpr size_t SMEM_RESULT = BM * sizeof(float);                                   // result accumulation
  constexpr size_t SMEM_SIZE = SMEM_DATA + SMEM_SF + SMEM_BARRIER + SMEM_RESULT;

  // launch cluster at runtime
  cudaLaunchConfig_t config = {0};
  config.gridDim = grid;
  config.blockDim = BLOCK_SIZE;
  config.dynamicSmemBytes = SMEM_SIZE;

  // Define the specific kernel specialization
  auto kernel = gemv_kernel<BK_PER_ITER, BK_ITER, BM_PER_ITER, BM_ITER, 
                            SFK, TK, STAGES, BLOCK_SIZE>;

  // Handle the attribute setting inside the single path
  if (SMEM_SIZE > 48 * 1024) 
  {
    CUDA_CHECK(cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      SMEM_SIZE
    ));
  }

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = 1;
  attribute[0].val.clusterDim.y = NUM_BLOCKS_IN_K;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  cudaLaunchKernelEx(&config, kernel,
                     reinterpret_cast<f16*>(c.data_ptr()),
                     B, M, K, N,
                     tensor_map_a, tensor_map_b, tensor_map_sfa, tensor_map_sfb);
}

template void gemv<512, 2, 8, 4>(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
template void gemv<512, 4, 8, 4>(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

gemv_cpp_src = """
#include <torch/extension.h>

template<const int BK_PER_ITER, const int BK_ITER, const int BM_PER_ITER, const int BM_ITER>
void gemv(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);

void gemv_k512x2(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c)
{
  gemv<512, 2, 8, 4>(a, b, sfa, sfb, c);
}

void gemv_k512x4(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c)
{
  gemv<512, 4, 8, 4>(a, b, sfa, sfb, c);
}
"""

gemv_module = load_inline(
    name='gemv_cuda',
    cpp_sources=gemv_cpp_src,
    cuda_sources=gemv_cuda_src,
    functions=['gemv_k512x2', 'gemv_k512x4'],
    verbose=True,
    extra_cuda_cflags=['-lineinfo', '-gencode=arch=compute_100a,code=sm_100a']
)

def gemv(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    gemv_module.gemv_k512x4(
            a,
            b,
            sfa,
            sfb,
            c,
        ) 
    
    # if a.size(1) * 2 <= 1024*8:
    #     gemv_module.gemv_k512x2(
    #         a,
    #         b,
    #         sfa,
    #         sfb,
    #         c,
    #     ) 
    # else:
    #     gemv_module.gemv_k512x4(
    #         a,
    #         b,
    #         sfa,
    #         sfb,
    #         c,
    #     )
  
def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemv
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float4e2m1fn] of shape [m, k, l],
            b: torch.Tensor[float4e2m1fn] of shape [1, k, l],
            sfa: torch.Tensor[float8_e4m3fnuz] of shape [m, k // 16, l], used by reference implementation
            sfb: torch.Tensor[float8_e4m3fnuz] of shape [1, k // 16, l], used by reference implementation
            sfa_permuted: torch.Tensor[float8_e4m3fnuz] of shape [32, 4, rest_m, 4, rest_k, l],
            sfb_permuted: torch.Tensor[float8_e4m3fnuz] of shape [32, 4, rest_n, 4, rest_k, l],
            c: torch.Tensor[float16] of shape [m, 1, l]
    Returns:
        Tensor containing output in float16
        c: torch.Tensor[float16] of shape [m, 1, l]
    """
    # c: [l, m, 1] is pre-allocated memory to avoid timing allocation overhead.
    a, b, sfa, sfb, sfa_permuted, sfb_permuted, c = data    

    gemv(a, b, sfa, sfb, c)

    return c
