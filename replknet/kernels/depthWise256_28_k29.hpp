#ifndef DEPTHWISE256_28_K29_HPP
#define DEPTHWISE256_28_K29_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive kernel (weights in shared memory, input from global/L2)
void depthWise_forward_ch_256_kernel_29_tensor_28_fp16_naive_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

// Column-merge 8×12 kernel (shared input loads across 3 adjacent pixels)
void depthWise_forward_ch_256_kernel_29_tensor_28_fp16_colmerge_8x12_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

#endif
