#ifndef DEPTHWISE128_56_K31_HPP
#define DEPTHWISE128_56_K31_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Chunk-based kernel (weights + input staging in shared memory)
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

// Naive kernel (weights in shared memory, input from global/L2)
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_naive_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

// Branchless kernel (4 separate tight loops, no branches in compute body)
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_branchless_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

// Row-merge kernel (shared row loads for pixel pairs)
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_rowmerge_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

// Row-merge 8×12 kernel (140 blocks for optimal SM occupancy)
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_rowmerge_8x12_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

// Column-merge 8×12 kernel (shared input loads across 3 adjacent pixels)
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_colmerge_8x12_C_interface(
    half* inputTensor, const half* weight, half* outputTensor,
    int tensorHeight, int tensorWidth, int Ch, int windowSize);

#endif
