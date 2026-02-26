/*
 * 29×29 Depthwise Convolution - Naive + Colmerge Kernels
 *
 * Target: RepLKNet-31B Stage 2
 * - Channels: 256
 * - Spatial: 28×28
 * - Kernel: 29×29
 * - Padding: 14 (same-padding)
 *
 * Naive: 512 threads, 4×4 warp layout, 4 pixels/warp, weights in shmem
 * Colmerge: 512 threads, 4×4 warp layout, 6 pixels/warp (8×12 tiles),
 *           zone-based column merge with 3 FMAs/load in main body
 */

#include "depthWise256_28_k29.hpp"
#include <iostream>

//=============================================================================
// NAIVE KERNEL: 29×29, weights in shared memory, input from global/L2
//=============================================================================
__global__ void depthWise_forward_ch_256_kernel_29_tensor_28_fp16_naive(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor)
{
    extern __shared__ half sharedMem[];
    half* weightCache = sharedMem;

    const int tid = threadIdx.x;
    const int laneId = tid & 31;
    const int warpIdx = tid >> 5;  // 0-15

    const int blockI = blockIdx.x * 8;
    const int blockJ = blockIdx.y * 8;
    const int blockCh = blockIdx.z * 32;

    const int H = 28, W = 28, C = 256;
    const int padSize = 14;

    // Load weights: 29*29*32 = 26912 half values, 512 threads
    for (int i = tid; i < 29 * 29 * 32; i += 512) {
        int kh = i / (29 * 32);
        int rem = i % (29 * 32);
        int kw = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (kh * 29 + kw) * C + blockCh + c_local;
        weightCache[i] = weight[globalIdx];
    }
    __syncthreads();

    // 4×4 warp layout
    int warpI = warpIdx / 4;  // 0-3
    int warpJ = warpIdx % 4;  // 0-3

    // 4 output pixels per warp
    int outI0 = blockI + warpI;       int outJ0 = blockJ + warpJ;
    int outI1 = blockI + warpI;       int outJ1 = blockJ + warpJ + 4;
    int outI2 = blockI + warpI + 4;   int outJ2 = blockJ + warpJ;
    int outI3 = blockI + warpI + 4;   int outJ3 = blockJ + warpJ + 4;

    half acc0 = __float2half(0.0f);
    half acc1 = __float2half(0.0f);
    half acc2 = __float2half(0.0f);
    half acc3 = __float2half(0.0f);

    int ch = blockCh + laneId;

    // 29×29 convolution loop
    for (int ki = 0; ki < 29; ++ki) {
        for (int kj = 0; kj < 29; ++kj) {
            half w = weightCache[(ki * 29 + kj) * 32 + laneId];

            // Pixel 0
            {
                int inI = outI0 + ki - padSize;
                int inJ = outJ0 + kj - padSize;
                if (inI >= 0 && inI < H && inJ >= 0 && inJ < W) {
                    half in_val = inputTensor[(inI * W + inJ) * C + ch];
                    acc0 = __hfma(in_val, w, acc0);
                }
            }
            // Pixel 1
            {
                int inI = outI1 + ki - padSize;
                int inJ = outJ1 + kj - padSize;
                if (inI >= 0 && inI < H && inJ >= 0 && inJ < W) {
                    half in_val = inputTensor[(inI * W + inJ) * C + ch];
                    acc1 = __hfma(in_val, w, acc1);
                }
            }
            // Pixel 2
            {
                int inI = outI2 + ki - padSize;
                int inJ = outJ2 + kj - padSize;
                if (inI >= 0 && inI < H && inJ >= 0 && inJ < W) {
                    half in_val = inputTensor[(inI * W + inJ) * C + ch];
                    acc2 = __hfma(in_val, w, acc2);
                }
            }
            // Pixel 3
            {
                int inI = outI3 + ki - padSize;
                int inJ = outJ3 + kj - padSize;
                if (inI >= 0 && inI < H && inJ >= 0 && inJ < W) {
                    half in_val = inputTensor[(inI * W + inJ) * C + ch];
                    acc3 = __hfma(in_val, w, acc3);
                }
            }
        }
    }

    // Write outputs (guard OOB)
    if (outI0 < H && outJ0 < W)
        outputTensor[(outI0 * W + outJ0) * C + ch] = acc0;
    if (outI1 < H && outJ1 < W)
        outputTensor[(outI1 * W + outJ1) * C + ch] = acc1;
    if (outI2 < H && outJ2 < W)
        outputTensor[(outI2 * W + outJ2) * C + ch] = acc2;
    if (outI3 < H && outJ3 < W)
        outputTensor[(outI3 * W + outJ3) * C + ch] = acc3;
}


//=============================================================================
// COLUMN-MERGE 8×12 KERNEL: Shared input loads across 3 column-adjacent pixels
// Grid: dim3(4, 3, 8) = 96 blocks
// 4×4 warp layout, 6 pixels/warp (2 rows × 3 columns, offset=4)
// Zone-based: z3 has 21 iterations with 3 FMAs/load
//=============================================================================
__global__ void depthWise_forward_ch_256_kernel_29_tensor_28_fp16_colmerge_8x12(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor)
{
    extern __shared__ half sharedMem[];
    half* weightCache = sharedMem;

    const int tid = threadIdx.x;
    const int laneId = tid & 31;
    const int warpIdx = tid >> 5;

    const int blockI = blockIdx.x * 8;
    const int blockJ = blockIdx.y * 12;
    const int blockCh = blockIdx.z * 32;

    const int H = 28, W = 28, C = 256;
    const int padSize = 14;

    // Load weights: 29*29*32 = 26912 half values
    for (int i = tid; i < 29 * 29 * 32; i += 512) {
        int kh = i / (29 * 32);
        int rem = i % (29 * 32);
        int kw = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (kh * 29 + kw) * C + blockCh + c_local;
        weightCache[i] = weight[globalIdx];
    }
    __syncthreads();

    int warpI = __shfl_sync(0xffffffff, warpIdx / 4, 0);
    int warpJ = __shfl_sync(0xffffffff, warpIdx % 4, 0);

    int outIA = blockI + warpI;
    int outIB = blockI + warpI + 4;
    int outJ0 = blockJ + warpJ;
    int outJ1 = blockJ + warpJ + 4;
    int outJ2 = blockJ + warpJ + 8;

    int ch = blockCh + laneId;

    half accA0 = __float2half(0.0f);
    half accA1 = __float2half(0.0f);
    half accA2 = __float2half(0.0f);
    half accB0 = __float2half(0.0f);
    half accB1 = __float2half(0.0f);
    half accB2 = __float2half(0.0f);

    // Compute zone bounds for column iteration
    // 3 pixels at columns outJ0, outJ1=outJ0+4, outJ2=outJ0+8
    // kj0 is the kernel-j index for pixel 0: inJ = kj0 + outJ0 - padSize
    // For pixel 1: kernel-j = kj0 - 4
    // For pixel 2: kernel-j = kj0 - 8
    int inJ_start = max(0, outJ0 - padSize);
    int inJ_end   = min(W, (outJ2 < W ? outJ2 + padSize + 1 : outJ0 + padSize + 1));
    int kj0_s_raw = inJ_start - outJ0 + padSize;
    int kj0_e_raw = inJ_end - outJ0 + padSize;

    // Zone bounds (offset=4, kernel=29):
    // z1: [0, 4)   — pixel 0 only
    // z2: [4, 8)   — pixels 0, 1
    // z3: [8, 29)  — all 3 pixels (21 iters, main body)
    // z4: [29, 33) — pixels 1, 2
    // z5: [33, 37) — pixel 2 only
    int z1_s = max(kj0_s_raw, 0);     int z1_e = min(4, min(kj0_e_raw, 29));
    int z2_s = max(kj0_s_raw, 4);     int z2_e = min(8, min(kj0_e_raw, 29));
    int z3_s = max(kj0_s_raw, 8);     int z3_e = min(29, kj0_e_raw);
    int z4_s = max(kj0_s_raw, 29);    int z4_e = min(33, kj0_e_raw);
    int z5_s = max(kj0_s_raw, 33);    int z5_e = min(37, kj0_e_raw);

    #define ZONE_BODY_ROW(ROW_OUT, ACC0, ACC1, ACC2) \
    { \
        int kiRow = ki * 29; \
        int rowBase = (ROW_OUT + ki - padSize) * W * C; \
        for (int kj0 = z1_s; kj0 < z1_e; ++kj0) { \
            half in_val = inputTensor[rowBase + (kj0 + outJ0 - padSize) * C + ch]; \
            ACC0 = __hfma(in_val, weightCache[(kiRow + kj0) * 32 + laneId], ACC0); \
        } \
        for (int kj0 = z2_s; kj0 < z2_e; ++kj0) { \
            half in_val = inputTensor[rowBase + (kj0 + outJ0 - padSize) * C + ch]; \
            ACC0 = __hfma(in_val, weightCache[(kiRow + kj0) * 32 + laneId], ACC0); \
            ACC1 = __hfma(in_val, weightCache[(kiRow + kj0 - 4) * 32 + laneId], ACC1); \
        } \
        for (int kj0 = z3_s; kj0 < z3_e; ++kj0) { \
            half in_val = inputTensor[rowBase + (kj0 + outJ0 - padSize) * C + ch]; \
            ACC0 = __hfma(in_val, weightCache[(kiRow + kj0) * 32 + laneId], ACC0); \
            ACC1 = __hfma(in_val, weightCache[(kiRow + kj0 - 4) * 32 + laneId], ACC1); \
            ACC2 = __hfma(in_val, weightCache[(kiRow + kj0 - 8) * 32 + laneId], ACC2); \
        } \
        for (int kj0 = z4_s; kj0 < z4_e; ++kj0) { \
            half in_val = inputTensor[rowBase + (kj0 + outJ0 - padSize) * C + ch]; \
            ACC1 = __hfma(in_val, weightCache[(kiRow + kj0 - 4) * 32 + laneId], ACC1); \
            ACC2 = __hfma(in_val, weightCache[(kiRow + kj0 - 8) * 32 + laneId], ACC2); \
        } \
        for (int kj0 = z5_s; kj0 < z5_e; ++kj0) { \
            half in_val = inputTensor[rowBase + (kj0 + outJ0 - padSize) * C + ch]; \
            ACC2 = __hfma(in_val, weightCache[(kiRow + kj0 - 8) * 32 + laneId], ACC2); \
        } \
    }

    // No interior block optimization: H=28 with padding=14 means no block
    // has full [0,29) ki range valid for both rows. Always use variable bounds.
    int kiA_s = max(0, padSize - outIA);  int kiA_e = min(29, H + padSize - outIA);
    int kiB_s = max(0, padSize - outIB);  int kiB_e = min(29, H + padSize - outIB);

    // Row A
    for (int ki = kiA_s; ki < kiA_e; ++ki) {
        ZONE_BODY_ROW(outIA, accA0, accA1, accA2)
    }
    // Row B (guard outIB < H)
    if (outIB < H) {
        for (int ki = kiB_s; ki < kiB_e; ++ki) {
            ZONE_BODY_ROW(outIB, accB0, accB1, accB2)
        }
    }

    #undef ZONE_BODY_ROW

    // Write outputs (guard OOB columns and rows)
    if (outJ0 < W) {
        outputTensor[(outIA * W + outJ0) * C + ch] = accA0;
        if (outIB < H)
            outputTensor[(outIB * W + outJ0) * C + ch] = accB0;
    }
    if (outJ1 < W) {
        outputTensor[(outIA * W + outJ1) * C + ch] = accA1;
        if (outIB < H)
            outputTensor[(outIB * W + outJ1) * C + ch] = accB1;
    }
    if (outJ2 < W) {
        outputTensor[(outIA * W + outJ2) * C + ch] = accA2;
        if (outIB < H)
            outputTensor[(outIB * W + outJ2) * C + ch] = accB2;
    }
}


//=============================================================================
// C Interface: Naive
//=============================================================================
void depthWise_forward_ch_256_kernel_29_tensor_28_fp16_naive_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 28 || tensorWidth != 28 || Ch != 256 || windowSize != 29) {
        std::cout << "error: expected 256ch, 28×28, kernel 29. Got "
                  << Ch << "ch, " << tensorHeight << "×" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(4, 4, 8);  // ceil(28/8)=4, 256/32=8, 128 blocks
    int threadSize = 512;
    size_t sharedMemSize = 29 * 29 * 32 * sizeof(half);

    cudaFuncSetAttribute(
        depthWise_forward_ch_256_kernel_29_tensor_28_fp16_naive,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_256_kernel_29_tensor_28_fp16_naive
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// C Interface: Column-merge 8×12
//=============================================================================
void depthWise_forward_ch_256_kernel_29_tensor_28_fp16_colmerge_8x12_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 28 || tensorWidth != 28 || Ch != 256 || windowSize != 29) {
        std::cout << "error: expected 256ch, 28×28, kernel 29. Got "
                  << Ch << "ch, " << tensorHeight << "×" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(4, 3, 8);  // ceil(28/8)=4, ceil(28/12)=3, 256/32=8 → 96 blocks
    int threadSize = 512;
    size_t sharedMemSize = 29 * 29 * 32 * sizeof(half);

    cudaFuncSetAttribute(
        depthWise_forward_ch_256_kernel_29_tensor_28_fp16_colmerge_8x12,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_256_kernel_29_tensor_28_fp16_colmerge_8x12
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}
