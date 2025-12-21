from task import input_t, output_t

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

# Kernel configuration parameters
ab_dtype = cutlass.Float4E2M1FN  # FP4 data type for A and B
sf_dtype = cutlass.Float8E4M3FN  # FP8 data type for scale factors
c_dtype = cutlass.Float16  # FP16 output type
sf_vec_size = 16  # Scale factor block size (16 elements share one scale)

def get_tiled_copies():
    """Create tiled copy objects within an MLIR context."""
    # TV layout
    thr_layout = cute.make_ordered_layout((1, 128), order=(1, 0)) 
    op = cute.nvgpu.CopyUniversalOp()

    ## For data
    val_layout_data = cute.make_ordered_layout((1, 32), order=(1, 0)) # Each thread handles 32 fp4
    atom_data = cute.make_copy_atom(op, ab_dtype, num_bits_per_copy=128)

    tiled_copy_data = cute.make_tiled_copy_tv(
        atom_data,
        thr_layout,
        val_layout_data
    )

    ## For scale factor
    val_layout_sf = cute.make_ordered_layout((1, 2), order=(1, 0)) # Each thread needs 2 sf
    atom_sf = cute.make_copy_atom(op, sf_dtype, num_bits_per_copy=32)

    tiled_copy_sf = cute.make_tiled_copy_tv(
        atom_sf,
        thr_layout,
        val_layout_sf
    )
    
    return thr_layout, tiled_copy_data, tiled_copy_sf

# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b

# The CuTe reference implementation for NVFP4 block-scaled GEMV
@cute.kernel
def kernel(
    mA_mkl: cute.Tensor,
    mB_nkl: cute.Tensor,
    mSFA_mkl: cute.Tensor,
    mSFB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
):
    # Get CUDA block and thread indices
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()

    thr_layout, tiled_copy_data, tiled_copy_sf = get_tiled_copies()

    # Use local_tile to select batch AND tile in one operation
    # For A and SFA
    # gA of shape: [(tile_M, tile_K), rest_M, rest_K, rest_L]
    gA = cute.tiled_divide(mA_mkl, tiled_copy_data.tiler_mn)
    gSFA = cute.tiled_divide(mSFA_mkl, tiled_copy_sf.tiler_mn)
    
    # For B and SFB
    gB = cute.tiled_divide(mB_nkl, tiled_copy_data.tiler_mn)
    gSFB = cute.tiled_divide(mSFB_nkl, tiled_copy_sf.tiler_mn)

    # For C
    # gC of shape: [(tile_M, tile_N), rest_M, rest_N, rest_L]
    gC = cute.tiled_divide(mC_mnl, (4, 1))
    
    if bidx == 0 and tidx == 0:
        cute.printf("gA: {}", gA)
        cute.printf("gSFA: {}", gSFA)
        cute.printf("gB: {}", gB)
        cute.printf("gSFB: {}", gSFB)
        cute.printf("gC: {}", gC)
    
    # # For C
    # gC = cute.local_tile(mC_mnl, (4, 1))

    # # gA shape: [tile_M, tile_K, rest_M, rest_K, rest_L]
    # tAgA = gA[warp_idx, tidx * 32:(tidx + 1) * 32, bidx, :, bidz]
    # tBgB = gB[0, tidx * 32:(tidx + 1) * 32, bidx, :, bidz]
    # tAgSFA = gSFA[warp_idx, tidx * 2:(tidx + 1) * 2, bidx, :, bidz]
    # tBgSFB = gSFB[0, tidx * 2:(tidx + 1) * 2, bidx, :, bidz]
    
    # # Output shape: [tile_M, tile_N, rest_M, rest_N, rest_L]
    # tCgC = gC[warp_idx, 0, bidx, 0, bidz]
    # tCgC = cute.make_tensor(tCgC.iterator, 1)
    # res = cute.zeros_like(tCgC, cutlass.Float32)

    # # if bidx == 0 and tidx % 32 == 0:
    # #     cute.printf("[{}] mA shape: {}, layout: {}", tidx, mA_mkl.shape, mA_mkl.layout)
    # #     cute.printf("[{}] tAgA shape: {}, layout: {}", tidx, tAgA.shape, tAgA.layout)
    #     # cute.printf("tAgSFA shape: {}, layout: {}", tAgSFA.shape, tAgSFA.layout)
    #     # cute.printf("tBgB shape: {}, layout: {}", tBgB.shape, tBgB.layout)
    #     # cute.printf("tBgSFB shape: {}, layout: {}", tBgSFB.shape, tBgSFB.layout)
    
    # tArA = cute.make_rmem_tensor_like(tAgA, ab_dtype)
    # tBrB = cute.make_rmem_tensor_like(tBgB, ab_dtype)
    # tArSFA = cute.make_rmem_tensor_like(tAgSFA, sf_dtype)
    # tBrSFB = cute.make_rmem_tensor_like(tBgSFB, sf_dtype)

    # cute.copy(tiled_copy_data, tAgA, tArA)
    # cute.copy(tiled_copy_data, tBgB, tBrB)
    # cute.copy(tiled_copy_sf, tAgSFA, tArSFA)
    # cute.copy(tiled_copy_sf, tBgSFB, tBrSFB)

    

    return


@cute.jit
def my_kernel(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
):
    """
    Host-side JIT function to prepare tensors and launch GPU kernel.
    """
    m, _, k, l = problem_size
    sf_k = k // sf_vec_size

    # Create CuTe Tensor via pointer and problem size.
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (m, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
        ),
    )

    sfa_tensor = cute.make_tensor(
        sfa_ptr,
        cute.make_layout(
            (m, cute.assume(sf_k, 16), l),
            stride=(cute.assume(sf_k, 16), 1, cute.assume(m * sf_k, 32)),
        ),
    )

    n_padded_128 = 128
    b_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (n_padded_128, cute.assume(k, 32), l),
            stride=(cute.assume(k, 32), 1, cute.assume(n_padded_128 * k, 32)),
        ),
    )

    sfb_tensor = cute.make_tensor(
        sfb_ptr,
        cute.make_layout(
            (n_padded_128, cute.assume(sf_k, 16), l),
            stride=(cute.assume(sf_k, 16), 1, cute.assume(n_padded_128 * sf_k, 32)),
        ),
    )

    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 32), 1, l), stride=(1, 1, m))
    )

    # Launch the CUDA kernel
    kernel(a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor).launch(
        grid=[c_tensor.shape[0], 1 , c_tensor.shape[2]],
        block=[128, 1, 1],
        cluster=(1, 1, 1),
    )

    return


# Global cache for compiled kernel
_compiled_kernel_cache = None

# This function is used to compile the kernel once and cache it and then allow users to
# run the kernel multiple times to get more accurate timing results.
def compile_kernel():
    """
    Compile the kernel once and cache it.
    This should be called before any timing measurements.

    Returns:
        The compiled kernel function
    """
    global _compiled_kernel_cache

    if _compiled_kernel_cache is not None:
        return _compiled_kernel_cache

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

    # Compile the kernel
    _compiled_kernel_cache = cute.compile(
        my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0)
    )

    return _compiled_kernel_cache


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled GEMV kernel.

    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.

    Args:
        data: Tuple of (a, b, sfa_cpu, sfb_cpu, c) PyTorch tensors
            a: [m, k, l] - Input matrix in float4e2m1fn
            b: [1, k, l] - Input vector in float4e2m1fn
            sfa: [m, k, l] - Scale factors in float8_e4m3fn
            sfb: [1, k, l] - Scale factors in float8_e4m3fn
            sfa_permuted: [32, 4, rest_m, 4, rest_k, l] - Scale factors in float8_e4m3fn
            sfb_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn
            c: [m, 1, l] - Output vector in float16

    Returns:
        Output tensor c with computed GEMV results
    """
    a, b, sfa, sfb, _, _, c = data

    # Ensure kernel is compiled (will use cached version if available)
    # To avoid the compilation overhead, we compile the kernel once and cache it.
    compiled_func = compile_kernel()

    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2
    # GEMV N dimension is always 1
    n = 1

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, sfa.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, sfb.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)

    # Execute the compiled kernel
    compiled_func(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l))

    return c
