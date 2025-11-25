#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

using Reg128 = uint4;
using Reg32 = uint32_t;
using Reg16 = uint16_t;
using f16 = __half;
using uint8 = uint8_t;
using fp8_e4m3 = __nv_fp8_e4m3;

// taken from https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/gemv_blockscaled.h#L564
__device__ __forceinline__ f16 blockscaled_multiply_add(
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
  const int BK,
  const int BM,
  const int WM,
  const int TK,
  const int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void gemv_kernel(
  const uint8* __restrict__ a,
  const uint8* __restrict__ b,
  const fp8_e4m3* __restrict__ sfa,
  const fp8_e4m3* __restrict__ sfb,
  f16* __restrict__ c,
  const int B,
  const int M,
  const int K,
  const int NUM_SF_BLOCKS_PER_K)
{
  // Each sf blocks stores scaling factors for 128x64 nvfp4 elemnts
  constexpr int SF_FP4_ROW = 128;
  constexpr int SF_FP4_COL = 64;

  assert(BM == SF_FP4_ROW); // BM must equal SF_FP4_ROW for now
  
  constexpr int SF_BLOCK_CNT = BK / SF_FP4_COL; // number of scaling blocks a block needs

  // Each sf block's layout is 32x4x4 fp8 elements
  constexpr int SF_BLOCK_HEIGHT = 32;
  constexpr int SF_BLOCK_WIDTH = 16;
  constexpr int SF_SIZE_PER_BLOCK = SF_BLOCK_HEIGHT * SF_BLOCK_WIDTH;

  __shared__ uint8 as[BM][BK/2];
  __shared__ fp8_e4m3 sfas[SF_BLOCK_CNT * SF_SIZE_PER_BLOCK];
  __shared__ uint8 bs[BK/2];
  __shared__ fp8_e4m3 sfbs[BK/16];

  const int tid = threadIdx.x;
  const int l = blockIdx.y;
  const int m_block = blockIdx.x * BM;
  const int brow = l * M + m_block;

  constexpr int THREADS_PER_K = BK / TK;
  const int tcol = (tid % THREADS_PER_K) * TK;
  const int trow = tid / THREADS_PER_K;

  a += (brow + trow) * K/2;
  b += l * K/2;
  c += brow + trow;

  // scaling factors are of shape (l, rest_m, rest_k, 32, 4, 4)
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x

  sfa += (brow / SF_FP4_ROW) * NUM_SF_BLOCKS_PER_K * SF_SIZE_PER_BLOCK;
  sfb += l * NUM_SF_BLOCKS_PER_K * SF_SIZE_PER_BLOCK;

  float accum = 0.0f; // accumulate on float to preserve precision

  for (int k = 0; k < K/BK; k++)
  {
    // load in data
    // all threads load A
    reinterpret_cast<Reg128*>(&as[trow][tcol / 2])[0] = 
      reinterpret_cast<const Reg128*>(&a[tcol / 2])[0];
    
    // threads from trow=0 load B 
    if (trow == 0)
    {
      reinterpret_cast<Reg128*>(&bs[tcol / 2])[0] = 
        reinterpret_cast<const Reg128*>(&b[tcol / 2])[0];
    }

    // some threads load 2 sf blocks of sfa
    // each thread can load 16 fp8 elements
    if (16 * tid < SF_BLOCK_CNT * SF_SIZE_PER_BLOCK)
    {
      reinterpret_cast<Reg128*>(&sfas[16 * tid])[0] = 
        reinterpret_cast<const Reg128*>(&sfa[16 * tid])[0];
    }

    // only a very few threads load sfb
    // we just need 4 fp8 out of the entire 128x4 sfb block
    if (4 * tid < BK / 16)
    {
      reinterpret_cast<Reg32*>(&sfbs[4 * tid])[0] = 
        reinterpret_cast<const Reg32*>(&sfb[tid * SF_SIZE_PER_BLOCK])[0]; // 4 elements at the start of the block
    }

    __syncthreads();

    a += BK/2;
    b += BK/2;
    sfa += SF_BLOCK_CNT * SF_SIZE_PER_BLOCK;
    sfb += SF_BLOCK_CNT * SF_SIZE_PER_BLOCK;

    // fma here
    Reg128 a_reg128 = reinterpret_cast<Reg128*>(&as[trow][tcol / 2])[0];
    Reg128 b_reg128 = reinterpret_cast<Reg128*>(&bs[tcol / 2])[0];
    Reg16 sfb_reg = reinterpret_cast<Reg16*>(&sfbs[tcol / 16])[0];

    Reg32* a_reg32_ptr = reinterpret_cast<Reg32*>(&a_reg128);
    Reg32* b_reg32_ptr = reinterpret_cast<Reg32*>(&b_reg128);

    // only tricky part is sfa
    const int sf_block_idx = tcol / SF_FP4_COL; // which of the 2 sf blocks
    const int sf_within_block_subcol = (trow / SF_BLOCK_HEIGHT); // which of the 4 subcolumns within the sf block
    const int sf_within_block_row = trow % SF_BLOCK_HEIGHT; // which of the 32 rows within the sf block
    const int sf_within_block_col = (tcol / 16) % 4; // which of the 16 columns within the sf block


    Reg16 sfa_reg = reinterpret_cast<Reg16*>(&sfas[sf_block_idx * SF_SIZE_PER_BLOCK
                                                   + sf_within_block_row * SF_BLOCK_WIDTH
                                                   + sf_within_block_subcol * 4
                                                   + sf_within_block_col
                                                  ])[0];

    if (M == 128 && K == 128 && B == 1 && tid < 4 && blockIdx.x == 0) {      
      uint8_t* a_bytes = reinterpret_cast<uint8_t*>(&a_reg128);
      printf("[%d]: A reg bytes: {%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d}\\n", 
        tid, 
        a_bytes[0], a_bytes[1], a_bytes[2], a_bytes[3],
        a_bytes[4], a_bytes[5], a_bytes[6], a_bytes[7],
        a_bytes[8], a_bytes[9], a_bytes[10], a_bytes[11],
        a_bytes[12], a_bytes[13], a_bytes[14], a_bytes[15]
      );

      printf("\\n");

      // print b registers in uint8
      uint8_t* b_bytes = reinterpret_cast<uint8_t*>(&b_reg128);
      printf("[%d]: B reg bytes: {%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d}\\n", 
        tid,
        b_bytes[0], b_bytes[1], b_bytes[2], b_bytes[3],
        b_bytes[4], b_bytes[5], b_bytes[6], b_bytes[7],
        b_bytes[8], b_bytes[9], b_bytes[10], b_bytes[11],
        b_bytes[12], b_bytes[13], b_bytes[14], b_bytes[15]
      );

      printf("\\n");

      // print sfa and sfb as 2 uint8
      uint8_t* sfa_bytes = reinterpret_cast<uint8_t*>(&sfa_reg);
      uint8_t* sfb_bytes = reinterpret_cast<uint8_t*>(&sfb_reg);

      printf("[%d]: sfa: %d %d\\n", tid, sfa_bytes[0], sfa_bytes[1]);
      printf("[%d]: sfb: %d %d\\n", tid, sfb_bytes[0], sfb_bytes[1]);
    }

    f16 res = blockscaled_multiply_add(
                a_reg32_ptr[0], a_reg32_ptr[1], a_reg32_ptr[2], a_reg32_ptr[3],
                b_reg32_ptr[0], b_reg32_ptr[1], b_reg32_ptr[2], b_reg32_ptr[3],
                sfa_reg,
                sfb_reg
              );

    accum +=  __half2float(res);
              
    if (M == 128 && K == 128 && B == 1 && tid < 4 && blockIdx.x == 0) {
      printf("[%d]: accum=%f, res=%f\\n", tid, accum, __half2float(res));
    }

    __syncthreads();
  }

  // reduce within threads on the same k
  #pragma unroll
  for (int offset = 1; offset < THREADS_PER_K; offset *= 2)
  {
    accum += __shfl_xor_sync(0xFFFFFFFF, accum, offset);
  }

  // write back result only by the first thread in each K
  if (tcol == 0)
  {
    c[trow] = __float2half_rn(accum);
  }
}


void gemv(
  torch::Tensor a, 
  torch::Tensor b,
  torch::Tensor sfa,
  torch::Tensor sfb,
  torch::Tensor c)
{
  const int M = a.size(0);
  const int K = a.size(1);
  const int B = a.size(2);

  const int NUM_SF_BLOCKS_PER_K = (K + 15) / 16;

  constexpr int TK = 32; // Each thread processes 32 elements along K dimension
  constexpr int BK = 128; // Each block processes 128 elements along K dimension
  constexpr int BM = 128; // Each block processes 128 elements along M dimension

  assert(M % BM == 0); // M must be divisible by BM
  assert(K % BK == 0); // K must be divisible by BK

  constexpr int THREADS_PER_K = BK / TK;
  constexpr int WM = 32 / THREADS_PER_K; // Number of rows per warp

  assert(BM % WM == 0); // BM must be divisible by WM

  constexpr int NUM_WARPS = BM / WM;
  constexpr int BLOCK_SIZE = NUM_WARPS * 32;
  
  const dim3 grid(M / BM, B);

  gemv_kernel<BK, BM, WM, TK, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(
    static_cast<uint8*>(a.data_ptr()),
    static_cast<uint8*>(b.data_ptr()),
    reinterpret_cast<fp8_e4m3*>(sfa.data_ptr<torch::Float8_e4m3fn>()),
    reinterpret_cast<fp8_e4m3*>(sfb.data_ptr<torch::Float8_e4m3fn>()),
    reinterpret_cast<f16*>(c.data_ptr<torch::Half>()),
    B,
    M,
    K,
    NUM_SF_BLOCKS_PER_K
  );

  cudaDeviceSynchronize();
}