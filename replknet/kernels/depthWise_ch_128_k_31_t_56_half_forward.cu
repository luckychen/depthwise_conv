/*
 * 31×31 Depthwise Convolution - Chunk-Based Shared Memory Kernel
 *
 * Target: RepLKNet-31B Stage 1
 * - Channels: 128
 * - Spatial: 56×56
 * - Kernel: 31×31
 * - Padding: 15 (same-padding)
 *
 * Design:
 * - 512 threads = 16 warps in 4×4 layout
 * - Each thread handles 1 FP16 channel (32 channels per block)
 * - Grid: (7, 7, 4) = 196 blocks
 * - Each warp computes 4 output pixels (8×8 tile per block)
 * - Weight in shared memory: 31×31×32 = 30,752 half (61,504 bytes) → dynamic shmem
 * - Input staging: 4×4×32 = 512 half (1,024 bytes)
 * - Total dynamic shared memory: 62,528 bytes
 * - Padded input region: 40×40, iterated as 10×10 chunks of 4×4 pixels
 */

#include "depthWise128_56_k31.hpp"
#include <iostream>

//=============================================================================
// NAIVE KERNEL: 31×31, weights in shared memory, input from global/L2
//=============================================================================
__global__ void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_naive(
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

    const int H = 56, W = 56, C = 128;
    const int padSize = 15;

    // Load weights: 31*31*32 = 30752 half values, 512 threads
    for (int i = tid; i < 31 * 31 * 32; i += 512) {
        int kh = i / (31 * 32);
        int rem = i % (31 * 32);
        int kw = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (kh * 31 + kw) * C + blockCh + c_local;
        weightCache[i] = weight[globalIdx];
    }
    __syncthreads();

    // 4×4 warp layout
    int warpI = warpIdx / 4;  // 0-3
    int warpJ = warpIdx % 4;  // 0-3

    // 4 output pixels per warp
    int outI0 = blockI + warpI;
    int outJ0 = blockJ + warpJ;
    int outI1 = blockI + warpI;
    int outJ1 = blockJ + warpJ + 4;
    int outI2 = blockI + warpI + 4;
    int outJ2 = blockJ + warpJ;
    int outI3 = blockI + warpI + 4;
    int outJ3 = blockJ + warpJ + 4;

    half acc0 = __float2half(0.0f);
    half acc1 = __float2half(0.0f);
    half acc2 = __float2half(0.0f);
    half acc3 = __float2half(0.0f);

    int ch = blockCh + laneId;

    // 31×31 convolution loop
    for (int ki = 0; ki < 31; ++ki) {
        for (int kj = 0; kj < 31; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];

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

    // Write outputs
    outputTensor[(outI0 * W + outJ0) * C + ch] = acc0;
    outputTensor[(outI1 * W + outJ1) * C + ch] = acc1;
    outputTensor[(outI2 * W + outJ2) * C + ch] = acc2;
    outputTensor[(outI3 * W + outJ3) * C + ch] = acc3;
}


//=============================================================================
// CHUNK-BASED KERNEL: 31×31, weights + input staging in dynamic shared memory
// Iterates 10×10 = 100 chunks of 4×4 pixels over the 40×40 padded input region
//=============================================================================
__global__ void __launch_bounds__(512, 1)
depthWise_forward_ch_128_kernel_31_tensor_56_fp16_chunk(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor)
{
    extern __shared__ half sharedMem[];
    half* weightCache  = sharedMem;                   // 31*31*32 = 30752 half
    half* inputStaging = sharedMem + 31 * 31 * 32;    // 4*4*32 = 512 half

    const int tid = threadIdx.x;
    const int laneId = tid & 31;
    const int warpIdx = tid >> 5;  // 0-15

    const int blockI = blockIdx.x * 8;
    const int blockJ = blockIdx.y * 8;
    const int blockCh = blockIdx.z * 32;

    const int H = 56, W = 56, C = 128;
    const int padSize = 15;

    // Use __shfl_sync to make warpI/warpJ warp-uniform
    int warpI = __shfl_sync(0xffffffff, warpIdx / 4, 0);
    int warpJ = __shfl_sync(0xffffffff, warpIdx % 4, 0);

    // Load weights: 30752 half values, 512 threads → 61 iterations
    for (int i = tid; i < 31 * 31 * 32; i += 512) {
        int kh = i / (31 * 32);
        int rem = i % (31 * 32);
        int kw = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (kh * 31 + kw) * C + blockCh + c_local;
        weightCache[i] = weight[globalIdx];
    }
    __syncthreads();

    // 4 output pixel positions
    int outI0 = blockI + warpI;       int outJ0 = blockJ + warpJ;
    int outI1 = blockI + warpI;       int outJ1 = blockJ + warpJ + 4;
    int outI2 = blockI + warpI + 4;   int outJ2 = blockJ + warpJ;
    int outI3 = blockI + warpI + 4;   int outJ3 = blockJ + warpJ + 4;

    half acc0 = __float2half(0.0f);
    half acc1 = __float2half(0.0f);
    half acc2 = __float2half(0.0f);
    half acc3 = __float2half(0.0f);

    // Thread mapping for chunk loading: 512 threads load 16 pixels × 32 channels
    int localPixel = tid / 32;   // 0-15: which pixel in 4×4
    int localCh    = tid % 32;   // 0-31: which channel

    int pixelRow = localPixel / 4;  // 0-3
    int pixelCol = localPixel % 4;  // 0-3

    // Padded input region starts at (blockI-16, blockJ-16), size 40×40
    // Aligned to multiple of 4: blockI-15 rounded down = blockI-16 if blockI%4==0
    // Actually: blockI is always multiple of 8, so blockI-16 is always multiple of 8
    int baseI = blockI - 16;
    int baseJ = blockJ - 16;

    // Iterate 10×10 = 100 chunks
    for (int cr = 0; cr < 10; ++cr) {
        for (int cc = 0; cc < 10; ++cc) {
            int chunkBaseI = baseI + cr * 4;
            int chunkBaseJ = baseJ + cc * 4;

            // Load chunk: 16 pixels × 32 channels = 512 values = 512 threads
            int globalI = chunkBaseI + pixelRow;
            int globalJ = chunkBaseJ + pixelCol;

            if (globalI >= 0 && globalI < H && globalJ >= 0 && globalJ < W) {
                inputStaging[localPixel * 32 + localCh] =
                    inputTensor[(globalI * W + globalJ) * C + blockCh + localCh];
            } else {
                inputStaging[localPixel * 32 + localCh] = __float2half(0.0f);
            }
            __syncthreads();

            // Compute: for each of 16 pixels in chunk, check contribution to each output
            for (int pr = 0; pr < 4; ++pr) {
                int gI = chunkBaseI + pr;
                // Early row skip: if this row can't contribute to any output
                // Output rows are outI0..outI3, input needs to be within [outI-15, outI+15]
                // Quick check: gI must be in range [blockI-15, blockI+7+15] = [blockI-15, blockI+22]
                if (gI < blockI - padSize || gI > blockI + 7 + padSize) continue;
                if (gI < 0 || gI >= H) continue;

                for (int pc = 0; pc < 4; ++pc) {
                    int gJ = chunkBaseJ + pc;
                    if (gJ < blockJ - padSize || gJ > blockJ + 7 + padSize) continue;
                    if (gJ < 0 || gJ >= W) continue;

                    half inputVal = inputStaging[(pr * 4 + pc) * 32 + laneId];

                    // Check contribution to pixel 0
                    {
                        int wI = gI - outI0 + padSize;
                        int wJ = gJ - outJ0 + padSize;
                        if (wI >= 0 && wI < 31 && wJ >= 0 && wJ < 31) {
                            half w = weightCache[(wI * 31 + wJ) * 32 + laneId];
                            acc0 = __hfma(inputVal, w, acc0);
                        }
                    }
                    // Check contribution to pixel 1
                    {
                        int wI = gI - outI1 + padSize;
                        int wJ = gJ - outJ1 + padSize;
                        if (wI >= 0 && wI < 31 && wJ >= 0 && wJ < 31) {
                            half w = weightCache[(wI * 31 + wJ) * 32 + laneId];
                            acc1 = __hfma(inputVal, w, acc1);
                        }
                    }
                    // Check contribution to pixel 2
                    {
                        int wI = gI - outI2 + padSize;
                        int wJ = gJ - outJ2 + padSize;
                        if (wI >= 0 && wI < 31 && wJ >= 0 && wJ < 31) {
                            half w = weightCache[(wI * 31 + wJ) * 32 + laneId];
                            acc2 = __hfma(inputVal, w, acc2);
                        }
                    }
                    // Check contribution to pixel 3
                    {
                        int wI = gI - outI3 + padSize;
                        int wJ = gJ - outJ3 + padSize;
                        if (wI >= 0 && wI < 31 && wJ >= 0 && wJ < 31) {
                            half w = weightCache[(wI * 31 + wJ) * 32 + laneId];
                            acc3 = __hfma(inputVal, w, acc3);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write outputs
    int ch = blockCh + laneId;
    outputTensor[(outI0 * W + outJ0) * C + ch] = acc0;
    outputTensor[(outI1 * W + outJ1) * C + ch] = acc1;
    outputTensor[(outI2 * W + outJ2) * C + ch] = acc2;
    outputTensor[(outI3 * W + outJ3) * C + ch] = acc3;
}


//=============================================================================
// BRANCHLESS KERNEL: 4 separate tight loops per pixel, zero branches in body
//=============================================================================
__global__ void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_branchless(
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
    const int blockJ = blockIdx.y * 8;
    const int blockCh = blockIdx.z * 32;

    const int H = 56, W = 56, C = 128;
    const int padSize = 15;

    // Load weights: 31*31*32 = 30752 half values
    for (int i = tid; i < 31 * 31 * 32; i += 512) {
        int kh = i / (31 * 32);
        int rem = i % (31 * 32);
        int kw = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (kh * 31 + kw) * C + blockCh + c_local;
        weightCache[i] = weight[globalIdx];
    }
    __syncthreads();

    int warpI = warpIdx / 4;
    int warpJ = warpIdx % 4;

    int outI0 = blockI + warpI;       int outJ0 = blockJ + warpJ;
    int outI1 = blockI + warpI;       int outJ1 = blockJ + warpJ + 4;
    int outI2 = blockI + warpI + 4;   int outJ2 = blockJ + warpJ;
    int outI3 = blockI + warpI + 4;   int outJ3 = blockJ + warpJ + 4;

    int ch = blockCh + laneId;

    half acc0 = __float2half(0.0f);
    half acc1 = __float2half(0.0f);
    half acc2 = __float2half(0.0f);
    half acc3 = __float2half(0.0f);

    // Pre-compute valid loop bounds for each pixel
    int ki0_s = max(0, padSize - outI0);  int ki0_e = min(31, H + padSize - outI0);
    int kj0_s = max(0, padSize - outJ0);  int kj0_e = min(31, W + padSize - outJ0);
    int ki1_s = max(0, padSize - outI1);  int ki1_e = min(31, H + padSize - outI1);
    int kj1_s = max(0, padSize - outJ1);  int kj1_e = min(31, W + padSize - outJ1);
    int ki2_s = max(0, padSize - outI2);  int ki2_e = min(31, H + padSize - outI2);
    int kj2_s = max(0, padSize - outJ2);  int kj2_e = min(31, W + padSize - outJ2);
    int ki3_s = max(0, padSize - outI3);  int ki3_e = min(31, H + padSize - outI3);
    int kj3_s = max(0, padSize - outJ3);  int kj3_e = min(31, W + padSize - outJ3);

    // Pixel 0: tight loops, zero branches in body
    for (int ki = ki0_s; ki < ki0_e; ++ki) {
        int rowBase = (outI0 + ki - padSize) * W * C;
        for (int kj = kj0_s; kj < kj0_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ0 + kj - padSize) * C + ch];
            acc0 = __hfma(in_val, w, acc0);
        }
    }

    // Pixel 1
    for (int ki = ki1_s; ki < ki1_e; ++ki) {
        int rowBase = (outI1 + ki - padSize) * W * C;
        for (int kj = kj1_s; kj < kj1_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ1 + kj - padSize) * C + ch];
            acc1 = __hfma(in_val, w, acc1);
        }
    }

    // Pixel 2
    for (int ki = ki2_s; ki < ki2_e; ++ki) {
        int rowBase = (outI2 + ki - padSize) * W * C;
        for (int kj = kj2_s; kj < kj2_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ2 + kj - padSize) * C + ch];
            acc2 = __hfma(in_val, w, acc2);
        }
    }

    // Pixel 3
    for (int ki = ki3_s; ki < ki3_e; ++ki) {
        int rowBase = (outI3 + ki - padSize) * W * C;
        for (int kj = kj3_s; kj < kj3_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ3 + kj - padSize) * C + ch];
            acc3 = __hfma(in_val, w, acc3);
        }
    }

    // Write outputs
    outputTensor[(outI0 * W + outJ0) * C + ch] = acc0;
    outputTensor[(outI1 * W + outJ1) * C + ch] = acc1;
    outputTensor[(outI2 * W + outJ2) * C + ch] = acc2;
    outputTensor[(outI3 * W + outJ3) * C + ch] = acc3;
}


//=============================================================================
// ROW-MERGE KERNEL: Shared row iteration for pixel pairs (0&1, 2&3)
//=============================================================================
__global__ void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_rowmerge(
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
    const int blockJ = blockIdx.y * 8;
    const int blockCh = blockIdx.z * 32;

    const int H = 56, W = 56, C = 128;
    const int padSize = 15;

    // Load weights
    for (int i = tid; i < 31 * 31 * 32; i += 512) {
        int kh = i / (31 * 32);
        int rem = i % (31 * 32);
        int kw = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (kh * 31 + kw) * C + blockCh + c_local;
        weightCache[i] = weight[globalIdx];
    }
    __syncthreads();

    int warpI = warpIdx / 4;
    int warpJ = warpIdx % 4;

    // Pixels 0&1 share outI01, pixels 2&3 share outI23
    int outI01 = blockI + warpI;
    int outJ0  = blockJ + warpJ;
    int outJ1  = blockJ + warpJ + 4;
    int outI23 = blockI + warpI + 4;
    int outJ2  = blockJ + warpJ;
    int outJ3  = blockJ + warpJ + 4;

    int ch = blockCh + laneId;

    half acc0 = __float2half(0.0f);
    half acc1 = __float2half(0.0f);
    half acc2 = __float2half(0.0f);
    half acc3 = __float2half(0.0f);

    // Row bounds for pixel pair 0&1 (same row)
    int ki01_s = max(0, padSize - outI01);  int ki01_e = min(31, H + padSize - outI01);
    int kj0_s  = max(0, padSize - outJ0);   int kj0_e  = min(31, W + padSize - outJ0);
    int kj1_s  = max(0, padSize - outJ1);   int kj1_e  = min(31, W + padSize - outJ1);

    // Row bounds for pixel pair 2&3 (same row)
    int ki23_s = max(0, padSize - outI23);  int ki23_e = min(31, H + padSize - outI23);
    int kj2_s  = max(0, padSize - outJ2);   int kj2_e  = min(31, W + padSize - outJ2);
    int kj3_s  = max(0, padSize - outJ3);   int kj3_e  = min(31, W + padSize - outJ3);

    // Pixel pair 0&1: shared row iteration
    for (int ki = ki01_s; ki < ki01_e; ++ki) {
        int rowBase = (outI01 + ki - padSize) * W * C;

        // Pixel 0 columns
        for (int kj = kj0_s; kj < kj0_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ0 + kj - padSize) * C + ch];
            acc0 = __hfma(in_val, w, acc0);
        }
        // Pixel 1 columns
        for (int kj = kj1_s; kj < kj1_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ1 + kj - padSize) * C + ch];
            acc1 = __hfma(in_val, w, acc1);
        }
    }

    // Pixel pair 2&3: shared row iteration
    for (int ki = ki23_s; ki < ki23_e; ++ki) {
        int rowBase = (outI23 + ki - padSize) * W * C;

        // Pixel 2 columns
        for (int kj = kj2_s; kj < kj2_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ2 + kj - padSize) * C + ch];
            acc2 = __hfma(in_val, w, acc2);
        }
        // Pixel 3 columns
        for (int kj = kj3_s; kj < kj3_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ3 + kj - padSize) * C + ch];
            acc3 = __hfma(in_val, w, acc3);
        }
    }

    // Write outputs
    outputTensor[(outI01 * W + outJ0) * C + ch] = acc0;
    outputTensor[(outI01 * W + outJ1) * C + ch] = acc1;
    outputTensor[(outI23 * W + outJ2) * C + ch] = acc2;
    outputTensor[(outI23 * W + outJ3) * C + ch] = acc3;
}


//=============================================================================
// ROW-MERGE 8×12 KERNEL: 140 blocks for near-perfect SM occupancy on 142 SMs
// 4×4 warp layout, each warp computes 6 pixels (2 rows × 3 columns)
// Row-merge: pixels in same row share ki loop, 3 separate kj loops
//=============================================================================
__global__ void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_rowmerge_8x12(
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

    const int H = 56, W = 56, C = 128;
    const int padSize = 15;

    // Load weights: 31*31*32 = 30752 half values
    for (int i = tid; i < 31 * 31 * 32; i += 512) {
        int kh = i / (31 * 32);
        int rem = i % (31 * 32);
        int kw = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (kh * 31 + kw) * C + blockCh + c_local;
        weightCache[i] = weight[globalIdx];
    }
    __syncthreads();

    // 4×4 warp layout over 8×12 tile
    // Use __shfl_sync to make warpI/warpJ warp-uniform (eliminates BSSY/BSYNC)
    int warpI = __shfl_sync(0xffffffff, warpIdx / 4, 0);  // 0-3
    int warpJ = __shfl_sync(0xffffffff, warpIdx % 4, 0);  // 0-3

    // 6 output pixels: 2 rows × 3 columns
    // Row A: (outIA, outJ0), (outIA, outJ1), (outIA, outJ2)
    // Row B: (outIB, outJ0), (outIB, outJ1), (outIB, outJ2)
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

    // Row bounds for row A
    int kiA_s = max(0, padSize - outIA);  int kiA_e = min(31, H + padSize - outIA);
    // Row bounds for row B
    int kiB_s = max(0, padSize - outIB);  int kiB_e = min(31, H + padSize - outIB);

    // Column bounds per pixel — OOB pixels get kj_s >= kj_e so loop body is skipped
    // No branches needed in the ki loop
    int kj0_s = max(0, padSize - outJ0);
    int kj0_e = (outJ0 < W) ? min(31, W + padSize - outJ0) : 0;
    int kj1_s = max(0, padSize - outJ1);
    int kj1_e = (outJ1 < W) ? min(31, W + padSize - outJ1) : 0;
    int kj2_s = max(0, padSize - outJ2);
    int kj2_e = (outJ2 < W) ? min(31, W + padSize - outJ2) : 0;

    // Row A: 3 pixels share ki loop, zero branches in body
    for (int ki = kiA_s; ki < kiA_e; ++ki) {
        int rowBase = (outIA + ki - padSize) * W * C;

        for (int kj = kj0_s; kj < kj0_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ0 + kj - padSize) * C + ch];
            accA0 = __hfma(in_val, w, accA0);
        }
        for (int kj = kj1_s; kj < kj1_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ1 + kj - padSize) * C + ch];
            accA1 = __hfma(in_val, w, accA1);
        }
        for (int kj = kj2_s; kj < kj2_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ2 + kj - padSize) * C + ch];
            accA2 = __hfma(in_val, w, accA2);
        }
    }

    // Row B: 3 pixels share ki loop, zero branches in body
    for (int ki = kiB_s; ki < kiB_e; ++ki) {
        int rowBase = (outIB + ki - padSize) * W * C;

        for (int kj = kj0_s; kj < kj0_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ0 + kj - padSize) * C + ch];
            accB0 = __hfma(in_val, w, accB0);
        }
        for (int kj = kj1_s; kj < kj1_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ1 + kj - padSize) * C + ch];
            accB1 = __hfma(in_val, w, accB1);
        }
        for (int kj = kj2_s; kj < kj2_e; ++kj) {
            half w = weightCache[(ki * 31 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ2 + kj - padSize) * C + ch];
            accB2 = __hfma(in_val, w, accB2);
        }
    }

    // Write outputs (guard OOB columns)
    if (outJ0 < W) {
        outputTensor[(outIA * W + outJ0) * C + ch] = accA0;
        outputTensor[(outIB * W + outJ0) * C + ch] = accB0;
    }
    if (outJ1 < W) {
        outputTensor[(outIA * W + outJ1) * C + ch] = accA1;
        outputTensor[(outIB * W + outJ1) * C + ch] = accB1;
    }
    if (outJ2 < W) {
        outputTensor[(outIA * W + outJ2) * C + ch] = accA2;
        outputTensor[(outIB * W + outJ2) * C + ch] = accB2;
    }
}


//=============================================================================
// COLUMN-MERGE 8×12 KERNEL: Shared input loads across 3 column-adjacent pixels
// For each ki row, iterate input columns once over the union range.
// Each input value is loaded once and applied to up to 3 pixels (different weights).
// Saves ~2.4× global input loads vs separate kj loops.
//=============================================================================
__global__ void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_colmerge_8x12(
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

    const int H = 56, W = 56, C = 128;
    const int padSize = 15;

    // Load weights
    for (int i = tid; i < 31 * 31 * 32; i += 512) {
        int kh = i / (31 * 32);
        int rem = i % (31 * 32);
        int kw = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (kh * 31 + kw) * C + blockCh + c_local;
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
    int inJ_start = max(0, outJ0 - padSize);
    int inJ_end   = min(W, (outJ2 < W ? outJ2 + padSize + 1 : outJ0 + padSize + 1));
    int kj0_s_raw = inJ_start - outJ0 + padSize;
    int kj0_e_raw = inJ_end - outJ0 + padSize;

    int z1_s = max(kj0_s_raw, 0);     int z1_e = min(4, min(kj0_e_raw, 31));
    int z2_s = max(kj0_s_raw, 4);     int z2_e = min(8, min(kj0_e_raw, 31));
    int z3_s = max(kj0_s_raw, 8);     int z3_e = min(31, kj0_e_raw);
    int z4_s = max(kj0_s_raw, 31);    int z4_e = min(35, kj0_e_raw);
    int z5_s = max(kj0_s_raw, 35);    int z5_e = min(39, kj0_e_raw);

    // Zone body macro: 5 zones per ki iteration
    // inJ = kj0 + outJ0 - padSize
    #define ZONE_BODY_ROW(ROW_OUT, ACC0, ACC1, ACC2) \
    { \
        int kiRow = ki * 31; \
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

    // Interior row blocks (blockIdx.x ∈ {2,3}): ki always [0, 31)
    // Compiler knows trip count = 31 at compile time → better scheduling
    // Boundary blocks: variable ki bounds from padding
    if (blockIdx.x >= 2 && blockIdx.x <= 3) {
        // Row A: ki ∈ [0, 31) — compile-time constant
        for (int ki = 0; ki < 31; ++ki) {
            ZONE_BODY_ROW(outIA, accA0, accA1, accA2)
        }
        // Row B: ki ∈ [0, 31) — compile-time constant
        for (int ki = 0; ki < 31; ++ki) {
            ZONE_BODY_ROW(outIB, accB0, accB1, accB2)
        }
    } else {
        int kiA_s = max(0, padSize - outIA);  int kiA_e = min(31, H + padSize - outIA);
        int kiB_s = max(0, padSize - outIB);  int kiB_e = min(31, H + padSize - outIB);
        // Row A: variable ki bounds
        for (int ki = kiA_s; ki < kiA_e; ++ki) {
            ZONE_BODY_ROW(outIA, accA0, accA1, accA2)
        }
        // Row B: variable ki bounds
        for (int ki = kiB_s; ki < kiB_e; ++ki) {
            ZONE_BODY_ROW(outIB, accB0, accB1, accB2)
        }
    }

    #undef ZONE_BODY_ROW

    // Write outputs
    if (outJ0 < W) {
        outputTensor[(outIA * W + outJ0) * C + ch] = accA0;
        outputTensor[(outIB * W + outJ0) * C + ch] = accB0;
    }
    if (outJ1 < W) {
        outputTensor[(outIA * W + outJ1) * C + ch] = accA1;
        outputTensor[(outIB * W + outJ1) * C + ch] = accB1;
    }
    if (outJ2 < W) {
        outputTensor[(outIA * W + outJ2) * C + ch] = accA2;
        outputTensor[(outIB * W + outJ2) * C + ch] = accB2;
    }
}


//=============================================================================
// C Interface: Column-merge 8×12
//=============================================================================
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_colmerge_8x12_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 56 || tensorWidth != 56 || Ch != 128 || windowSize != 31) {
        std::cout << "error: expected 128ch, 56×56, kernel 31. Got "
                  << Ch << "ch, " << tensorHeight << "×" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(7, 5, 4);  // 140 blocks
    int threadSize = 512;
    size_t sharedMemSize = 31 * 31 * 32 * sizeof(half);

    cudaFuncSetAttribute(
        depthWise_forward_ch_128_kernel_31_tensor_56_fp16_colmerge_8x12,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_128_kernel_31_tensor_56_fp16_colmerge_8x12
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// C Interface: Naive
//=============================================================================
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_naive_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 56 || tensorWidth != 56 || Ch != 128 || windowSize != 31) {
        std::cout << "error: expected 128ch, 56×56, kernel 31. Got "
                  << Ch << "ch, " << tensorHeight << "×" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(7, 7, 4);  // ceil(56/8)=7, 128/32=4
    int threadSize = 512;

    // Shared memory: weights only = 31*31*32 half = 61504 bytes
    size_t sharedMemSize = 31 * 31 * 32 * sizeof(half);

    // Need to set max dynamic shared memory since 61504 > 48KB default
    cudaFuncSetAttribute(
        depthWise_forward_ch_128_kernel_31_tensor_56_fp16_naive,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_128_kernel_31_tensor_56_fp16_naive
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// C Interface: Chunk-based
//=============================================================================
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 56 || tensorWidth != 56 || Ch != 128 || windowSize != 31) {
        std::cout << "error: expected 128ch, 56×56, kernel 31. Got "
                  << Ch << "ch, " << tensorHeight << "×" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(7, 7, 4);  // ceil(56/8)=7, 128/32=4
    int threadSize = 512;

    // Shared memory: weights (61504) + input staging (1024) = 62528 bytes
    size_t sharedMemSize = (31 * 31 * 32 + 4 * 4 * 32) * sizeof(half);

    cudaFuncSetAttribute(
        depthWise_forward_ch_128_kernel_31_tensor_56_fp16_chunk,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_128_kernel_31_tensor_56_fp16_chunk
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// C Interface: Branchless
//=============================================================================
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_branchless_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 56 || tensorWidth != 56 || Ch != 128 || windowSize != 31) {
        std::cout << "error: expected 128ch, 56×56, kernel 31. Got "
                  << Ch << "ch, " << tensorHeight << "×" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(7, 7, 4);
    int threadSize = 512;
    size_t sharedMemSize = 31 * 31 * 32 * sizeof(half);

    cudaFuncSetAttribute(
        depthWise_forward_ch_128_kernel_31_tensor_56_fp16_branchless,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_128_kernel_31_tensor_56_fp16_branchless
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// C Interface: Row-merge
//=============================================================================
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_rowmerge_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 56 || tensorWidth != 56 || Ch != 128 || windowSize != 31) {
        std::cout << "error: expected 128ch, 56×56, kernel 31. Got "
                  << Ch << "ch, " << tensorHeight << "×" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(7, 7, 4);
    int threadSize = 512;
    size_t sharedMemSize = 31 * 31 * 32 * sizeof(half);

    cudaFuncSetAttribute(
        depthWise_forward_ch_128_kernel_31_tensor_56_fp16_rowmerge,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_128_kernel_31_tensor_56_fp16_rowmerge
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// C Interface: Row-merge 8×12
//=============================================================================
void depthWise_forward_ch_128_kernel_31_tensor_56_fp16_rowmerge_8x12_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 56 || tensorWidth != 56 || Ch != 128 || windowSize != 31) {
        std::cout << "error: expected 128ch, 56×56, kernel 31. Got "
                  << Ch << "ch, " << tensorHeight << "×" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(7, 5, 4);  // ceil(56/8)=7, ceil(56/12)=5, 128/32=4 → 140 blocks
    int threadSize = 512;
    size_t sharedMemSize = 31 * 31 * 32 * sizeof(half);

    cudaFuncSetAttribute(
        depthWise_forward_ch_128_kernel_31_tensor_56_fp16_rowmerge_8x12,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_128_kernel_31_tensor_56_fp16_rowmerge_8x12
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}
