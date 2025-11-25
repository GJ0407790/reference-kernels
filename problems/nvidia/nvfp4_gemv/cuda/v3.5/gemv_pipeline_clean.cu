#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>

using Reg128 = uint4;
using Reg32 = uint32_t;
using Reg16 = uint16_t;
using f16 = __half;
using uint8 = uint8_t;
using fp8_e4m3 = __nv_fp8_e4m3;

// src contains 8 fp4 values
// convert to 4 f16x2 values
__device__ __forceinline__ void cvt_rn_f16x2_e2m1x2(
  const Reg32& src,
  Reg32& dst0, Reg32& dst1, Reg32& dst2, Reg32& dst3
)
{
  asm volatile(
    "{\\n" \\
    ".reg .b8 byte0, byte1, byte2, byte3;\\n" \\
    "mov.b32 {byte0, byte1, byte2, byte3}, %4;\\n" \\
    "cvt.rn.f16x2.e2m1x2 %0, byte0;\\n" \\
    "cvt.rn.f16x2.e2m1x2 %1, byte1;\\n" \\
    "cvt.rn.f16x2.e2m1x2 %2, byte2;\\n" \\
    "cvt.rn.f16x2.e2m1x2 %3, byte3;\\n" \\
    "}\\n"
    : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3) // 0, 1, 2, 3
    : "r"(src) // 4
    : "memory"
  );
}

__device__ __forceinline__ void cvt_rn_f16x2_e4m3x2(const Reg16& src, Reg32& dst)
{
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(dst) :"h"(src)
               : "memory");
}

// modify from https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/gemv_blockscaled.h#L564
__device__ __forceinline__ f16 blockscaled_multiply_add(
  const Reg32& a0, const Reg32& a1, const Reg32& a2, const Reg32& a3,
  const Reg32& b0, const Reg32& b1, const Reg32& b2, const Reg32& b3,
  const Reg32& b4, const Reg32& b5, const Reg32& b6, const Reg32& b7,
  const Reg32& b8, const Reg32& b9, const Reg32& b10, const Reg32& b11,
  const Reg32& b12, const Reg32& b13, const Reg32& b14, const Reg32& b15,
  const Reg16& sfa, const Reg32& sfb
)
{
  f16 res;
  Reg16* res_u16_ptr = reinterpret_cast<Reg16*>(&res); 

  asm volatile( \\
    "{\\n" \\
    // declare registers for A / B tensors
    ".reg .b8 byte0_0, byte0_1, byte0_2, byte0_3;\\n" \\
    ".reg .b8 byte1_0, byte1_1, byte1_2, byte1_3;\\n" \\
    ".reg .b8 byte2_0, byte2_1, byte2_2, byte2_3;\\n" \\
    ".reg .b8 byte3_0, byte3_1, byte3_2, byte3_3;\\n" \\

    // declare registers for accumulators
    ".reg .f16x2 accum_0_0, accum_0_1, accum_0_2, accum_0_3;\\n" \\
    ".reg .f16x2 accum_1_0, accum_1_1, accum_1_2, accum_1_3;\\n" \\
    ".reg .f16x2 accum_2_0, accum_2_1, accum_2_2, accum_2_3;\\n" \\
    ".reg .f16x2 accum_3_0, accum_3_1, accum_3_2, accum_3_3;\\n" \\

    // declare registers for scaling factors
    ".reg .f16x2 sfa_f16x2;\\n" \\
    ".reg .f16x2 sf_f16x2;\\n" \\

    // declare registers for conversion
    ".reg .f16x2 cvt_0_0, cvt_0_1, cvt_0_2, cvt_0_3;\\n" \\
    ".reg .f16x2 cvt_1_0, cvt_1_1, cvt_1_2, cvt_1_3;\\n" \\
    ".reg .f16x2 cvt_2_0, cvt_2_1, cvt_2_2, cvt_2_3;\\n" \\
    ".reg .f16x2 cvt_3_0, cvt_3_1, cvt_3_2, cvt_3_3;\\n" \\
    ".reg .f16 result_f16, lane0, lane1;\\n" \\
    ".reg .f16x2 mul_f16x2_0, mul_f16x2_1;\\n" \\

    // convert scaling factors from fp8 to f16x2
    "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %1;\\n" \\
    
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
    "mul.rn.f16x2 sf_f16x2, sfa_f16x2, %2;\\n" \\
    "mov.b32 {lane0, lane1}, sf_f16x2;\\n" \\
    "mov.b32 mul_f16x2_0, {lane0, lane0};\\n" \\
    "mov.b32 mul_f16x2_1, {lane1, lane1};\\n" \\

    // unpacking A tensors
    "mov.b32 {byte0_0, byte0_1, byte0_2, byte0_3}, %3;\\n" \\
    "mov.b32 {byte1_0, byte1_1, byte1_2, byte1_3}, %4;\\n" \\
    "mov.b32 {byte2_0, byte2_1, byte2_2, byte2_3}, %5;\\n" \\
    "mov.b32 {byte3_0, byte3_1, byte3_2, byte3_3}, %6;\\n" \\

    // convert A and B tensors from fp4 to f16x2

    // A[0 - 7]
    "cvt.rn.f16x2.e2m1x2 cvt_0_0, byte0_0;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_0_1, byte0_1;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_0_2, byte0_2;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_0_3, byte0_3;\\n" \\

    // A[8 - 15] and B[8 - 15]
    "cvt.rn.f16x2.e2m1x2 cvt_1_0, byte1_0;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_1_1, byte1_1;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_1_2, byte1_2;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_1_3, byte1_3;\\n" \\

    // A[16 - 23]
    "cvt.rn.f16x2.e2m1x2 cvt_2_0, byte2_0;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_2_1, byte2_1;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_2_2, byte2_2;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_2_3, byte2_3;\\n" \\

    // A[24 - 31]
    "cvt.rn.f16x2.e2m1x2 cvt_3_0, byte3_0;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_3_1, byte3_1;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_3_2, byte3_2;\\n" \\
    "cvt.rn.f16x2.e2m1x2 cvt_3_3, byte3_3;\\n" \\

    // fma for A[0 - 7] and B[0 - 7]
    "fma.rn.f16x2 accum_0_0, cvt_0_0, %7, accum_0_0;\\n" \\
    "fma.rn.f16x2 accum_0_1, cvt_0_1, %8, accum_0_1;\\n" \\
    "fma.rn.f16x2 accum_0_2, cvt_0_2, %9, accum_0_2;\\n" \\
    "fma.rn.f16x2 accum_0_3, cvt_0_3, %10, accum_0_3;\\n" \\

    // fma for A[8 - 15] and B[8 - 15]
    "fma.rn.f16x2 accum_1_0, cvt_1_0, %11, accum_1_0;\\n" \\
    "fma.rn.f16x2 accum_1_1, cvt_1_1, %12, accum_1_1;\\n" \\
    "fma.rn.f16x2 accum_1_2, cvt_1_2, %13, accum_1_2;\\n" \\
    "fma.rn.f16x2 accum_1_3, cvt_1_3, %14, accum_1_3;\\n" \\

    // fma for A[16 - 23] and B[16 - 23]
    "fma.rn.f16x2 accum_2_0, cvt_2_0, %15, accum_2_0;\\n" \\
    "fma.rn.f16x2 accum_2_1, cvt_2_1, %16, accum_2_1;\\n" \\
    "fma.rn.f16x2 accum_2_2, cvt_2_2, %17, accum_2_2;\\n" \\
    "fma.rn.f16x2 accum_2_3, cvt_2_3, %18, accum_2_3;\\n" \\

    // fma for A[24 - 31] and B[24 - 31]
    "fma.rn.f16x2 accum_3_0, cvt_3_0, %19, accum_3_0;\\n" \\
    "fma.rn.f16x2 accum_3_1, cvt_3_1, %20, accum_3_1;\\n" \\
    "fma.rn.f16x2 accum_3_2, cvt_3_2, %21, accum_3_2;\\n" \\
    "fma.rn.f16x2 accum_3_3, cvt_3_3, %22, accum_3_3;\\n" \\

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
    : "=h"(res_u16_ptr[0])                    // 0
    : "h"(sfa), "r"(sfb),                     // 1, 2
      "r"(a0), "r"(a1), "r"(a2), "r"(a3),     // 3, 4, 5, 6
      "r"(b0), "r"(b1), "r"(b2), "r"(b3),     // 7, 8, 9, 10
      "r"(b4), "r"(b5), "r"(b6), "r"(b7),     // 11, 12, 13, 14
      "r"(b8), "r"(b9), "r"(b10), "r"(b11),   // 15, 16, 17, 18
      "r"(b12), "r"(b13), "r"(b14), "r"(b15)  // 19, 20, 21, 22
    : "memory"
  );

  return res;
}

template<const int BM, const int TK, const int STAGES>
__global__ void gemv_kernel_safe(
  const uint8* __restrict__ a,
  const uint8* __restrict__ b,
  const fp8_e4m3* __restrict__ sfa,
  const fp8_e4m3* __restrict__ sfb,
  f16* __restrict__ c,
  const int B,
  const int M,
  const int K,
  const int N)
{
  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int l = blockIdx.y;
  const int global_row = l * M + blockIdx.x * BM;
  const int tcol = tid * TK;

  // static shmem for result
  __shared__ float result[BM];
  extern __shared__ char dyn_smem[];

  uint8* a_smem = reinterpret_cast<uint8*>(dyn_smem);
  fp8_e4m3* sfa_smem = reinterpret_cast<fp8_e4m3*>(a_smem + STAGES * (K / 2));

  #pragma unroll
  for (int i = tid; i < BM; i += blockDim.x)
  {
    result[i] = 0.0f;
  }

  // incremement global pointers
  a += (global_row * K + tcol) / 2;
  b += (l * N * K + tcol) / 2;
  c += global_row;
  sfa += (global_row * K + tcol) / 16;
  sfb += (l * N * K + tcol) / 16;

  // increment smem pointers
  a_smem += tcol / 2;
  sfa_smem += tcol / 16;

  float accum[BM] = {0.0f}; // accumulate on float to preserve precision

  Reg32 b_reg[TK/2]; // need this many registers to hold 32 fp16 values (converted from fp4)
  Reg32 sfb_reg; // need 1 32bit register to hold 2 fp16 values (converted from fp8)

  __syncthreads();
  
  if (tcol < K)
  {
    // load in b and sfb
    // convert once at the start
    Reg128 b_reg128 = reinterpret_cast<const Reg128*>(b)[0];
    Reg16 sfb_reg16 = reinterpret_cast<const Reg16*>(sfb)[0];

    Reg32* b_reg32_ptr = reinterpret_cast<Reg32*>(&b_reg128);

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
      cvt_rn_f16x2_e2m1x2(
        b_reg32_ptr[i],
        b_reg[4*i], b_reg[4*i+1], b_reg[4*i+2], b_reg[4*i+3]
      );
    }
    
    cvt_rn_f16x2_e4m3x2(sfb_reg16, sfb_reg);

    for (int compute_batch = 0, fetch_batch = 0; compute_batch < BM; compute_batch++)
    {
      // prefetch A and sfa into smem
      for (; fetch_batch < BM && fetch_batch < (compute_batch + STAGES); fetch_batch++)
      {
        const int shared_idx = fetch_batch % STAGES;

        __pipeline_memcpy_async(&a_smem[shared_idx * (K / 2)], a, 16); // load in 16B

        if (lane_id % 2 == 0) // minimally load in 4B, hence only even lanes load in sfa
        {
          __pipeline_memcpy_async(&sfa_smem[shared_idx * (K / 16)], sfa, 4);
        }

        __pipeline_commit();

        a += K / 2;
        sfa += K / 16;
      }

      __pipeline_wait_prior(fetch_batch - compute_batch - 1);
      __syncwarp(); // for sfa

      const int shared_idx = compute_batch % STAGES;
      Reg128 a_reg = reinterpret_cast<const Reg128*>(&a_smem[shared_idx * (K / 2)])[0];
      Reg16 sfa_reg = reinterpret_cast<const Reg16*>(&sfa_smem[shared_idx * (K / 16)])[0];

      // fma here
      Reg32* a_reg32_ptr = reinterpret_cast<Reg32*>(&a_reg);
      Reg32* b_reg32_ptr = reinterpret_cast<Reg32*>(&b_reg);

      f16 res = blockscaled_multiply_add(
                  a_reg32_ptr[0], a_reg32_ptr[1], a_reg32_ptr[2], a_reg32_ptr[3],
                  b_reg[0], b_reg[1], b_reg[2], b_reg[3],
                  b_reg[4], b_reg[5], b_reg[6], b_reg[7],
                  b_reg[8], b_reg[9], b_reg[10], b_reg[11],
                  b_reg[12], b_reg[13], b_reg[14], b_reg[15],
                  sfa_reg, sfb_reg
                );

      accum[compute_batch] +=  __half2float(res);


    }
  }
                
  // reduce within threads on the same k
  #pragma unroll
  for (int i = 0; i < BM; i++)
  {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
    {
      accum[i] += __shfl_xor_sync(0xFFFFFFFF, accum[i], offset);
    }
  }

  #pragma unroll
  for (int i = lane_id; i < BM; i += 32)
  {
    atomicAdd(&result[i], accum[i]);
  }

  __syncthreads();

  // write back result only by the first thread in each K
  #pragma unroll
  for (int i = tid; i < BM; i += blockDim.x)
  {
    c[i] = __float2half_rn(result[i]);
  }
}

void gemv_safe(
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

  constexpr int TK = 32; // Each thread processes 32 elements along K dimension
  constexpr int BM = 8;
  constexpr int STAGES = 4;

  assert(M % BM == 0); // M must be divisible by BM

  const int NUM_THREADS = K / TK;
  
  const dim3 grid(M / BM, B);

  const size_t SMEM_SIZE = STAGES * (K / 2 + K / 16); // a + sfa

  if (SMEM_SIZE > 48 * 1024) 
  {
    cudaFuncSetAttribute(
      gemv_kernel_safe<BM, TK, STAGES>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      SMEM_SIZE
    );
  }

  gemv_kernel_safe<BM, TK, STAGES><<<grid, NUM_THREADS, SMEM_SIZE>>>(
    static_cast<uint8*>(a.data_ptr()),
    static_cast<uint8*>(b.data_ptr()),
    reinterpret_cast<fp8_e4m3*>(sfa.data_ptr<torch::Float8_e4m3fn>()),
    reinterpret_cast<fp8_e4m3*>(sfb.data_ptr<torch::Float8_e4m3fn>()),
    reinterpret_cast<f16*>(c.data_ptr<torch::Half>()),
    B,
    M,
    K,
    N
  );
}

template<
  const int BM_PER_ITER,
  const int BM_ITER,
  const int TK, 
  const int STAGES>
__global__ void gemv_kernel_perf(
  const uint8* __restrict__ a,
  const uint8* __restrict__ b,
  const fp8_e4m3* __restrict__ sfa,
  const fp8_e4m3* __restrict__ sfb,
  f16* __restrict__ c,
  const int B,
  const int M,
  const int K,
  const int N)
{
  constexpr int BM = BM_PER_ITER * BM_ITER;
  const int tid = threadIdx.x;
  const int lane_id = tid % 32;
  const int l = blockIdx.y;
  const int global_row = l * M + blockIdx.x * BM;
  const int tcol = tid * TK;

  // static shmem for result
  __shared__ float result[BM];
  extern __shared__ char dyn_smem[];

  uint8* a_smem = reinterpret_cast<uint8*>(dyn_smem);
  fp8_e4m3* sfa_smem = reinterpret_cast<fp8_e4m3*>(a_smem + STAGES * (K / 2));

  #pragma unroll
  for (int i = tid; i < BM; i += blockDim.x)
  {
    result[i] = 0.0f;
  }

  // incremement global pointers
  a += (global_row * K + tcol) / 2;
  b += (l * N * K + tcol) / 2;
  c += global_row;
  sfa += (global_row * K + tcol) / 16;
  sfb += (l * N * K + tcol) / 16;

  // increment smem pointers
  a_smem += tcol / 2;
  sfa_smem += tcol / 16;

  Reg32 b_reg[TK/2]; // need this many registers to hold 32 fp16 values (converted from fp4)
  Reg32 sfb_reg; // need 1 32bit register to hold 2 fp16 values (converted from fp8)

  __syncthreads();

  // load in b and sfb
  // convert once at the start
  Reg128 b_reg128 = reinterpret_cast<const Reg128*>(b)[0];
  Reg16 sfb_reg16 = reinterpret_cast<const Reg16*>(sfb)[0];

  Reg32* b_reg32_ptr = reinterpret_cast<Reg32*>(&b_reg128);

  #pragma unroll
  for (int i = 0; i < 4; i++)
  {
    cvt_rn_f16x2_e2m1x2(
      b_reg32_ptr[i],
      b_reg[4*i], b_reg[4*i+1], b_reg[4*i+2], b_reg[4*i+3]
    );
  }
  
  cvt_rn_f16x2_e4m3x2(sfb_reg16, sfb_reg);

  // start the inner loop
  for (int bm = 0; bm < BM_ITER; bm++)
  {
    float accum[BM_PER_ITER] = {0.0f}; // accumulate on float to preserve precision

    for (int compute_batch = 0, fetch_batch = 0; compute_batch < BM_PER_ITER; compute_batch++)
    {
      // prefetch A and sfa into smem
      for (; fetch_batch < BM_PER_ITER && fetch_batch < (compute_batch + STAGES); fetch_batch++)
      {
        const int shared_idx = fetch_batch % STAGES;

        __pipeline_memcpy_async(&a_smem[shared_idx * (K / 2)], a, 16); // load in 16B

        if (lane_id % 2 == 0) // minimally load in 4B, hence only even lanes load in sfa
        {
          __pipeline_memcpy_async(&sfa_smem[shared_idx * (K / 16)], sfa, 4);
        }

        __pipeline_commit();

        a += K / 2;
        sfa += K / 16;
      }

      __pipeline_wait_prior(fetch_batch - compute_batch - 1);
      __syncwarp(); // for sfa

      const int shared_idx = compute_batch % STAGES;
      Reg128 a_reg = reinterpret_cast<const Reg128*>(&a_smem[shared_idx * (K / 2)])[0];
      Reg16 sfa_reg = reinterpret_cast<const Reg16*>(&sfa_smem[shared_idx * (K / 16)])[0];

      // fma here
      Reg32* a_reg32_ptr = reinterpret_cast<Reg32*>(&a_reg);
      Reg32* b_reg32_ptr = reinterpret_cast<Reg32*>(&b_reg);

      f16 res = blockscaled_multiply_add(
                  a_reg32_ptr[0], a_reg32_ptr[1], a_reg32_ptr[2], a_reg32_ptr[3],
                  b_reg[0], b_reg[1], b_reg[2], b_reg[3],
                  b_reg[4], b_reg[5], b_reg[6], b_reg[7],
                  b_reg[8], b_reg[9], b_reg[10], b_reg[11],
                  b_reg[12], b_reg[13], b_reg[14], b_reg[15],
                  sfa_reg, sfb_reg
                );

      accum[compute_batch] +=  __half2float(res);
    }
                  
    // reduce within threads on the same k
    #pragma unroll
    for (int i = 0; i < BM_PER_ITER; i++)
    {
      #pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1)
      {
        accum[i] += __shfl_xor_sync(0xFFFFFFFF, accum[i], offset);
      }
    }

    const int result_offset = bm * BM_PER_ITER;
    #pragma unroll
    for (int i = lane_id; i < BM_PER_ITER; i += 32)
    {
      atomicAdd(&result[result_offset + i], accum[i]);
    }
  }

  __syncthreads();
  // write back result only by the first thread in each K
  #pragma unroll
  for (int i = tid; i < BM; i += blockDim.x)
  {
    c[i] = __float2half_rn(result[i]);
  }
}


template<const int BM_PER_ITER, const int BM_ITER, const int STAGES>
void gemv_perf(
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

  constexpr int TK = 32; // Each thread processes 32 elements along K dimension
  constexpr int BM = BM_PER_ITER * BM_ITER;

  assert(M % BM == 0); // M must be divisible by BM

  const int NUM_THREADS = K / TK;
  
  const dim3 grid(M / BM, B);

  const size_t SMEM_SIZE = STAGES * (K / 2 + K / 16); // a + sfa

  if (SMEM_SIZE > 48 * 1024) 
  {
    cudaFuncSetAttribute(
      gemv_kernel_perf<BM_PER_ITER, BM_ITER, TK, STAGES>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      SMEM_SIZE
    );
  }

  gemv_kernel_perf<BM_PER_ITER, BM_ITER, TK, STAGES><<<grid, NUM_THREADS, SMEM_SIZE>>>(
    static_cast<uint8*>(a.data_ptr()),
    static_cast<uint8*>(b.data_ptr()),
    reinterpret_cast<fp8_e4m3*>(sfa.data_ptr<torch::Float8_e4m3fn>()),
    reinterpret_cast<fp8_e4m3*>(sfb.data_ptr<torch::Float8_e4m3fn>()),
    reinterpret_cast<f16*>(c.data_ptr<torch::Half>()),
    B,
    M,
    K,
    N
  );
}

template void gemv_perf<8, 4, 8>(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);