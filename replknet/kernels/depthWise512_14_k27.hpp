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

#endif
