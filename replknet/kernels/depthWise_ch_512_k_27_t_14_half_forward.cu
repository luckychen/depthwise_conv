/*
 * 27×27 Depthwise Convolution - FP16 NHWC
 *
 * Target configuration:
 * - Channels: 512
 * - Spatial: 14×14
 * - Kernel: 27×27
 * - Padding: 13 (same-padding)
 *
 * Key property: kernel (27) > spatial (14), so every output pixel's
 * receptive field covers the ENTIRE 14×14 input. The entire input
 * fits in shared memory alongside weights.
 *
 * Naive kernel:
 * - Grid: 2×7×16 = 224 blocks (7 rows × 2 cols per block)
 * - 512 threads = 16 warps
 * - 14 active warps (1 pixel each), 2 idle
 * - Weight in shared memory: 27×27×32 = 23,328 half (46,656 bytes)
 *
 * All-shared kernel:
 * - Grid: 2×7×16 = 224 blocks
 * - 256 threads = 8 warps
 * - 7 active warps (2 pixels each via colmerge), 1 idle
 * - Weight: 27×27×32 = 23,328 half (46,656 bytes)
 * - Input: 14×14×32 = 6,272 half (12,544 bytes)
 * - Total shared memory: 59,200 bytes
 * - Pure shared memory compute — zero global loads in hot loop
 */

#include "depthWise512_14_k27.hpp"
#include <iostream>


//=============================================================================
// NAIVE KERNEL: 27×27, weights in shared memory, input from global/L2
// Grid: 2×7×16, 512 threads, 14 active warps (1 pixel each)
//=============================================================================
__global__ void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor)
{
    extern __shared__ half sharedMem[];
    half* weightCache = sharedMem;

    const int tid = threadIdx.x;
    const int laneId = tid & 31;
    const int warpIdx = tid >> 5;  // 0-15

    const int blockI = blockIdx.x * 7;
    const int blockJ = blockIdx.y * 2;
    const int blockCh = blockIdx.z * 32;

    const int H = 14, W = 14, C = 512;
    const int padSize = 13;

    // Load weights: 27*27*32 = 23328 half values, 512 threads
    for (int i = tid; i < 27 * 27 * 32; i += 512) {
        int kh = i / (27 * 32);
        int rem = i % (27 * 32);
        int kw = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (kh * 27 + kw) * C + blockCh + c_local;
        weightCache[i] = weight[globalIdx];
    }
    __syncthreads();

    // 14 warps active, 2 idle
    if (warpIdx >= 14) return;

    // Pixel assignment: warpIdx / 2 → row, warpIdx % 2 → col
    int outI = blockI + (warpIdx / 2);
    int outJ = blockJ + (warpIdx % 2);

    int ch = blockCh + laneId;

    half acc = __float2half(0.0f);

    // Pre-compute valid loop bounds
    int ki_s = max(0, padSize - outI);
    int ki_e = min(27, H + padSize - outI);
    int kj_s = max(0, padSize - outJ);
    int kj_e = min(27, W + padSize - outJ);

    for (int ki = ki_s; ki < ki_e; ++ki) {
        int inI = outI + ki - padSize;
        int rowBase = inI * W * C;
        for (int kj = kj_s; kj < kj_e; ++kj) {
            half w = weightCache[(ki * 27 + kj) * 32 + laneId];
            half in_val = inputTensor[rowBase + (outJ + kj - padSize) * C + ch];
            acc = __hfma(in_val, w, acc);
        }
    }

    // Write output
    if (outI < H && outJ < W) {
        outputTensor[(outI * W + outJ) * C + ch] = acc;
    }
}


//=============================================================================
// ALL-SHARED KERNEL: weights + entire 14×14 input in shared memory
// Grid: 2×7×16, 256 threads, 7 active warps with colmerge (2 pixels each)
// Pure shared memory compute — zero global loads in hot loop
//=============================================================================
__global__ void __launch_bounds__(256, 1)
depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor)
{
    extern __shared__ half sharedMem[];
    half* weightCache = sharedMem;                    // 27*27*32 = 23328 half
    half* inputCache  = sharedMem + 27 * 27 * 32;    // 14*14*32 = 6272 half

    const int tid = threadIdx.x;
    const int laneId = tid & 31;
    const int warpIdx = tid >> 5;  // 0-7

    const int blockI = blockIdx.x * 7;
    const int blockJ = blockIdx.y * 2;
    const int blockCh = blockIdx.z * 32;

    const int H = 14, W = 14, C = 512;
    const int padSize = 13;

    // Phase 1: Load weights (23328 elements, 256 threads → 92 rounds)
    for (int i = tid; i < 27 * 27 * 32; i += 256) {
        int kh = i / (27 * 32);
        int rem = i % (27 * 32);
        int kw = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (kh * 27 + kw) * C + blockCh + c_local;
        weightCache[i] = weight[globalIdx];
    }

    // Phase 2: Load entire input (6272 elements, 256 threads → 25 rounds)
    for (int i = tid; i < 14 * 14 * 32; i += 256) {
        int row = i / (14 * 32);
        int rem = i % (14 * 32);
        int col = rem / 32;
        int c_local = rem % 32;
        int globalIdx = (row * W + col) * C + blockCh + c_local;
        inputCache[i] = inputTensor[globalIdx];
    }
    __syncthreads();

    // 7 warps active, 1 idle
    if (warpIdx >= 7) return;

    // Each warp handles one row in the 7-row block, 2 columns (colmerge)
    int outI = blockI + warpIdx;
    int outJ0 = blockJ;       // first column
    int outJ1 = blockJ + 1;   // second column (offset = 1)

    half acc0 = __float2half(0.0f);
    half acc1 = __float2half(0.0f);

    // Pre-compute ki bounds
    int ki_s = max(0, padSize - outI);
    int ki_e = min(27, H + padSize - outI);

    // Pre-compute kj bounds for pixel 0 (referenced as kj0)
    int kj0_s = max(0, padSize - outJ0);
    int kj0_e = min(27, W + padSize - outJ0);

    // Column merge zones (offset = outJ1 - outJ0 = 1):
    // kj0 is the weight column index for pixel 0
    // For pixel 1: weight column = kj0 - 1 (since outJ1 = outJ0 + 1)
    // Zone 1: kj0 ∈ [kj0_s, min(1, kj0_e)) — pixel 0 only (kj0-1 < 0 for pixel 1)
    // Zone 2: kj0 ∈ [max(kj0_s, 1), min(27, kj0_e)) — both pixels
    // Zone 3: kj0 ∈ [max(kj0_s, 27), kj0_e) — pixel 1 only (kj0 >= 27 for pixel 0)

    int z1_s = kj0_s;
    int z1_e = min(1, kj0_e);
    int z2_s = max(kj0_s, 1);
    int z2_e = min(27, kj0_e);
    int z3_s = max(kj0_s, 27);
    int z3_e = kj0_e;

    // Make bounds warp-uniform
    z1_s = __shfl_sync(0xffffffff, z1_s, 0);
    z1_e = __shfl_sync(0xffffffff, z1_e, 0);
    z2_s = __shfl_sync(0xffffffff, z2_s, 0);
    z2_e = __shfl_sync(0xffffffff, z2_e, 0);
    z3_s = __shfl_sync(0xffffffff, z3_s, 0);
    z3_e = __shfl_sync(0xffffffff, z3_e, 0);
    ki_s = __shfl_sync(0xffffffff, ki_s, 0);
    ki_e = __shfl_sync(0xffffffff, ki_e, 0);

    for (int ki = ki_s; ki < ki_e; ++ki) {
        int kiRow = ki * 27;
        int inI = outI + ki - padSize;
        int inRowBase = inI * 14 * 32;

        // Zone 1: pixel 0 only (at most 1 iteration)
        for (int kj0 = z1_s; kj0 < z1_e; ++kj0) {
            int inJ = outJ0 + kj0 - padSize;
            half in_val = inputCache[inRowBase + inJ * 32 + laneId];
            acc0 = __hfma(in_val, weightCache[(kiRow + kj0) * 32 + laneId], acc0);
        }

        // Zone 2: both pixels (main body, up to 26 iterations, 2 FMAs/load)
        for (int kj0 = z2_s; kj0 < z2_e; ++kj0) {
            int inJ = outJ0 + kj0 - padSize;
            half in_val = inputCache[inRowBase + inJ * 32 + laneId];
            acc0 = __hfma(in_val, weightCache[(kiRow + kj0) * 32 + laneId], acc0);
            acc1 = __hfma(in_val, weightCache[(kiRow + kj0 - 1) * 32 + laneId], acc1);
        }

        // Zone 3: pixel 1 only (at most 1 iteration)
        for (int kj0 = z3_s; kj0 < z3_e; ++kj0) {
            int inJ = outJ0 + kj0 - padSize;
            half in_val = inputCache[inRowBase + inJ * 32 + laneId];
            acc1 = __hfma(in_val, weightCache[(kiRow + kj0 - 1) * 32 + laneId], acc1);
        }
    }

    // Write outputs
    int ch = blockCh + laneId;
    if (outI < H && outJ0 < W) {
        outputTensor[(outI * W + outJ0) * C + ch] = acc0;
    }
    if (outI < H && outJ1 < W) {
        outputTensor[(outI * W + outJ1) * C + ch] = acc1;
    }
}


//=============================================================================
// NAIVE 64CH KERNEL: 27×27, 64 channels/block, half2 vectorization
// Grid: 2×7×8 = 112 blocks (single wave on 142 SMs)
// 512 threads = 16 warps, 14 active (1 pixel each), 2 idle
// Each lane handles 2 channels via half2
// Weight shmem: 27×27×64 = 46,656 half2 = 93,312 bytes (fits 100KB)
//=============================================================================
__global__ void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive64(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor)
{
    extern __shared__ half sharedMem[];
    half2* weightCache2 = reinterpret_cast<half2*>(sharedMem);

    const int tid = threadIdx.x;
    const int laneId = tid & 31;
    const int warpIdx = tid >> 5;  // 0-15

    const int blockI = blockIdx.x * 7;
    const int blockJ = blockIdx.y * 2;
    const int blockCh = blockIdx.z * 64;  // 64 channels per block

    const int H = 14, W = 14, C = 512;
    const int padSize = 13;

    // Load weights: 27*27*64 half = 46656 values = 23328 half2
    // 512 threads, each loads ~46 half2 values
    const int weightSize2 = 27 * 27 * 32;  // 23328 half2
    half2* weightGlobal2 = reinterpret_cast<half2*>(const_cast<half*>(weight));
    for (int i = tid; i < weightSize2; i += 512) {
        int kh = i / (27 * 32);
        int rem = i % (27 * 32);
        int kw = rem / 32;
        int c2 = rem % 32;  // half2 index within 64 channels
        // Global weight layout: [kh, kw, C] as half, read as half2
        int globalIdx2 = (kh * 27 + kw) * (C / 2) + (blockCh / 2) + c2;
        weightCache2[i] = weightGlobal2[globalIdx2];
    }
    __syncthreads();

    // 14 warps active, 2 idle
    if (warpIdx >= 14) return;

    int outI = blockI + (warpIdx / 2);
    int outJ = blockJ + (warpIdx % 2);

    // Each lane handles 2 channels via half2
    int ch2 = (blockCh / 2) + laneId;  // half2 offset in global tensor

    half2 acc = __float2half2_rn(0.0f);

    int ki_s = max(0, padSize - outI);
    int ki_e = min(27, H + padSize - outI);
    int kj_s = max(0, padSize - outJ);
    int kj_e = min(27, W + padSize - outJ);

    half2* inputTensor2 = reinterpret_cast<half2*>(inputTensor);

    for (int ki = ki_s; ki < ki_e; ++ki) {
        int inI = outI + ki - padSize;
        int rowBase2 = inI * W * (C / 2);
        for (int kj = kj_s; kj < kj_e; ++kj) {
            half2 w = weightCache2[(ki * 27 + kj) * 32 + laneId];
            half2 in_val = inputTensor2[rowBase2 + (outJ + kj - padSize) * (C / 2) + ch2];
            acc = __hfma2(in_val, w, acc);
        }
    }

    // Write output
    if (outI < H && outJ < W) {
        half2* outputTensor2 = reinterpret_cast<half2*>(outputTensor);
        outputTensor2[(outI * W + outJ) * (C / 2) + ch2] = acc;
    }
}


//=============================================================================
// ALL-SHARED BRANCH-FREE KERNEL (Cb=32, half2, split-warp colmerge)
// Fits within 99KB shmem limit on both Ada and GB10.
//
// K=27 > H=14: every input pixel contributes to every output — branch-free.
//
// Split-warp: each 32-thread warp → two 16-lane half-warps, each handling
// one output pixel via half2 (16 × 2ch = 32ch). Adjacent columns share
// input loads (broadcast) with different weight columns. Zero bank conflicts.
//
// Grid: (1, 7, 16) = 112 blocks (same as naive64 — single wave on GB10).
// 512 threads = 16 warps, 14 active (one per row), 2 idle.
// Shmem: weights 27×27×32 + input 14×14×32 = 59,200 bytes.
// Inner loop: branch-free 14×14, pure shmem reads, half2 FMAs.
//=============================================================================
__global__ void __launch_bounds__(512, 1)
depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_branchfree_h2_cb32(
    half* __restrict__ inputTensor,
    const half* __restrict__ weight,
    half* __restrict__ outputTensor)
{
    extern __shared__ half sharedMem[];
    half2* wCache2 = reinterpret_cast<half2*>(sharedMem);
    half2* iCache2 = reinterpret_cast<half2*>(sharedMem + 27 * 27 * 32);

    const int tid = threadIdx.x;
    const int laneId = tid & 31;
    const int warpIdx = tid >> 5;  // 0-15

    const int blockJ = blockIdx.y * 2;
    const int blockCh = blockIdx.z * 32;

    // Cooperative load: weights (11,664 half2, 512 threads → ~23 rounds)
    half2* wGlobal2 = reinterpret_cast<half2*>(const_cast<half*>(weight));
    for (int i = tid; i < 27 * 27 * 16; i += 512) {
        int kh = i / (27 * 16);
        int rem = i % (27 * 16);
        int kw = rem / 16;
        int c2 = rem % 16;
        wCache2[i] = wGlobal2[(kh * 27 + kw) * 256 + blockCh / 2 + c2];
    }

    // Cooperative load: input (3,136 half2, 512 threads → ~7 rounds)
    half2* iGlobal2 = reinterpret_cast<half2*>(inputTensor);
    for (int i = tid; i < 14 * 14 * 16; i += 512) {
        int row = i / (14 * 16);
        int rem = i % (14 * 16);
        int col = rem / 16;
        int c2 = rem % 16;
        iCache2[i] = iGlobal2[(row * 14 + col) * 256 + blockCh / 2 + c2];
    }
    __syncthreads();

    if (warpIdx >= 14) return;

    int outI = warpIdx;  // one row per warp, all 14 rows
    int localLane = laneId & 15;
    int myOutJ = blockJ + (laneId >> 4);

    half2 acc = __float2half2_rn(0.0f);

    #pragma unroll 2
    for (int inI = 0; inI < 14; ++inI) {
        int ki = inI - outI + 13;
        int kiBase = ki * 27 * 16;
        int inRowBase = inI * 14 * 16;

        #pragma unroll 2
        for (int inJ = 0; inJ < 14; ++inJ) {
            half2 iv = iCache2[inRowBase + inJ * 16 + localLane];
            int kj = inJ - myOutJ + 13;
            half2 wv = wCache2[kiBase + kj * 16 + localLane];
            acc = __hfma2(iv, wv, acc);
        }
    }

    half2* out2 = reinterpret_cast<half2*>(outputTensor);
    int ch2 = blockCh / 2 + localLane;
    out2[(outI * 14 + myOutJ) * 256 + ch2] = acc;
}


//=============================================================================
// C Interface: All-shared branch-free Cb=32 half2 split-warp colmerge
//=============================================================================
void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_branchfree_h2_cb32_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 14 || tensorWidth != 14 || Ch != 512 || windowSize != 27) {
        std::cout << "error: expected 512ch, 14x14, kernel 27. Got "
                  << Ch << "ch, " << tensorHeight << "x" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(1, 7, 16);  // 14 rows/block, ceil(14/2)=7, 512/32=16 → 112 blocks
    int threadSize = 512;
    size_t sharedMemSize = (27 * 27 * 32 + 14 * 14 * 32) * sizeof(half);  // 59200 bytes

    cudaFuncSetAttribute(
        depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_branchfree_h2_cb32,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_branchfree_h2_cb32
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// C Interface: Naive
//=============================================================================
void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 14 || tensorWidth != 14 || Ch != 512 || windowSize != 27) {
        std::cout << "error: expected 512ch, 14x14, kernel 27. Got "
                  << Ch << "ch, " << tensorHeight << "x" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(2, 7, 16);  // ceil(14/7)=2, ceil(14/2)=7, 512/32=16
    int threadSize = 512;
    size_t sharedMemSize = 27 * 27 * 32 * sizeof(half);  // 46656 bytes

    cudaFuncSetAttribute(
        depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// C Interface: All-shared colmerge
//=============================================================================
void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 14 || tensorWidth != 14 || Ch != 512 || windowSize != 27) {
        std::cout << "error: expected 512ch, 14x14, kernel 27. Got "
                  << Ch << "ch, " << tensorHeight << "x" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(2, 7, 16);  // ceil(14/7)=2, ceil(14/2)=7, 512/32=16
    int threadSize = 256;
    // weights (46656) + input (12544) = 59200 bytes
    size_t sharedMemSize = (27 * 27 * 32 + 14 * 14 * 32) * sizeof(half);

    cudaFuncSetAttribute(
        depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// C Interface: Naive 64ch/block with half2
//=============================================================================
void depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive64_C_interface(
    half* inputTensor,
    const half* weight,
    half* outputTensor,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize)
{
    if (tensorHeight != 14 || tensorWidth != 14 || Ch != 512 || windowSize != 27) {
        std::cout << "error: expected 512ch, 14x14, kernel 27. Got "
                  << Ch << "ch, " << tensorHeight << "x" << tensorWidth
                  << ", kernel " << windowSize << std::endl;
        exit(1);
    }

    dim3 gridSize(2, 7, 8);  // ceil(14/7)=2, ceil(14/2)=7, 512/64=8 → 112 blocks
    int threadSize = 512;
    size_t sharedMemSize = 27 * 27 * 64 * sizeof(half);  // 93312 bytes

    cudaFuncSetAttribute(
        depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive64,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive64
        <<<gridSize, threadSize, sharedMemSize>>>(inputTensor, weight, outputTensor);
}
