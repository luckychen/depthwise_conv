/*
 * RepLKNet Depthwise Kernels — PyTorch NHWC Integration Header
 *
 * Declares the __global__ kernel functions defined in the standalone .cu files
 * so the integration .cu can launch them on PyTorch's CUDA stream.
 */

#ifndef DEPTHWISE_REPLKNET_NHWC_INTEGRATION_HPP
#define DEPTHWISE_REPLKNET_NHWC_INTEGRATION_HPP

#include <cuda_fp16.h>

// 31×31, 128ch, 56×56 — colmerge 8×12
__global__ void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_colmerge_8x12(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor);

// 29×29, 256ch, 28×28 — colmerge 8×12
__global__ void depthWise_forward_ch_256_kernel_29_tensor_28_fp16_colmerge_8x12(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor);

// 27×27, 512ch, 14×14 — naive64 half2
__global__ void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive64(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor);

// 27×27, 512ch, 14×14 — allshared branchfree Cb=32 half2 split-warp colmerge
__global__ void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_branchfree_h2_cb32(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor);

#endif
