#ifndef DEPTHWISE512_14_K27_HPP
#define DEPTHWISE512_14_K27_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive kernel (weights in shared memory, input from global/L2)
void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

// All-shared-memory kernel (weights + entire input in shared memory, colmerge)
void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

// Naive kernel with 64ch/block and half2 vectorization (112 blocks, single wave)
void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive64_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

// All-shared branch-free Cb=32, half2 split-warp colmerge (both platforms, 59KB shmem)
__global__ void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_branchfree_h2_cb32(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor);

void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_branchfree_h2_cb32_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

#endif
