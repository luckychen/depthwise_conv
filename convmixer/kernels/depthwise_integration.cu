
#include "depthwise_integration.hpp"
#include <iostream>
#include <c10/cuda/CUDAStream.h>

//=============================================================================
// Integration kernels for PyTorch custom ops
// These are the production-ready kernels for model inference
//=============================================================================

//=============================================================================
// 32×32×768, 7×7 kernel (NAIVE - best for this configuration)
// Using 128ch/block variant for better SM utilization
//=============================================================================
__global__ void depthwise_32_768_7_kernel(half* inputTensor,
                            const half* weight,
							half* outputTensor)
{
	extern __shared__ half sharedMem[];
	half* weightCache = sharedMem;

	const int tid = threadIdx.x;
	const int warpLane = tid & 31;
	const int warpIdx = tid >> 5;

	const int blockI = blockIdx.x * 8;
	const int blockJ = blockIdx.y * 8;
	const int blockCh = blockIdx.z * 128;  // 128 channels per block

	const int chOffsetA = warpLane * 2;
	const int chOffsetB = 64 + warpLane * 2;

	const half2* weightGlobal = reinterpret_cast<const half2*>(weight);
	half2* weightCacheH2 = reinterpret_cast<half2*>(weightCache);

	// Load weights: 7*7*128 half = 7*7*64 half2 = 3136 half2
	// Phase 1-3: ALL 1024 threads
	for (int phase = 0; phase < 3; ++phase) {
		int loadIdx = tid + phase * 1024;
		int ki = loadIdx / (7 * 64);
		int rem = loadIdx % (7 * 64);
		int kj = rem / 64;
		int chIdx = rem % 64;
		int globalIdx = (ki * 7 + kj) * 768 + blockCh + chIdx * 2;
		weightCacheH2[loadIdx] = weightGlobal[globalIdx / 2];
	}
	// Phase 4: ONLY 64 threads
	if (tid < 64) {
		int loadIdx = tid + 3072;
		int ki = loadIdx / (7 * 64);
		int rem = loadIdx % (7 * 64);
		int kj = rem / 64;
		int chIdx = rem % 64;
		int globalIdx = (ki * 7 + kj) * 768 + blockCh + chIdx * 2;
		weightCacheH2[loadIdx] = weightGlobal[globalIdx / 2];
	}

	__syncthreads();

	int warpRow = warpIdx / 8;
	int warpCol = warpIdx % 8;

	int outI1 = blockI + warpRow;
	int outJ1 = blockJ + warpCol;
	int outI2 = blockI + warpRow + 4;
	int outJ2 = blockJ + warpCol;

	half2 acc1A = __float2half2_rn(0.0f);
	half2 acc1B = __float2half2_rn(0.0f);
	half2 acc2A = __float2half2_rn(0.0f);
	half2 acc2B = __float2half2_rn(0.0f);

	const half2* inputH2 = reinterpret_cast<const half2*>(inputTensor);
	const int padSize = 3;

	for (int ki = 0; ki < 7; ++ki) {
		for (int kj = 0; kj < 7; ++kj) {
			int wCacheBase = (ki * 7 + kj) * 64;
			half2 wA = weightCacheH2[wCacheBase + warpLane];
			half2 wB = weightCacheH2[wCacheBase + 32 + warpLane];

			int inI1 = outI1 + ki - padSize;
			int inJ1 = outJ1 + kj - padSize;
			if (inI1 >= 0 && inI1 < 32 && inJ1 >= 0 && inJ1 < 32) {
				int inBase = (inI1 * 32 + inJ1) * 768 + blockCh;
				half2 in1A = inputH2[(inBase + chOffsetA) / 2];
				half2 in1B = inputH2[(inBase + chOffsetB) / 2];
				acc1A = __hfma2(in1A, wA, acc1A);
				acc1B = __hfma2(in1B, wB, acc1B);
			}

			int inI2 = outI2 + ki - padSize;
			int inJ2 = outJ2 + kj - padSize;
			if (inI2 >= 0 && inI2 < 32 && inJ2 >= 0 && inJ2 < 32) {
				int inBase = (inI2 * 32 + inJ2) * 768 + blockCh;
				half2 in2A = inputH2[(inBase + chOffsetA) / 2];
				half2 in2B = inputH2[(inBase + chOffsetB) / 2];
				acc2A = __hfma2(in2A, wA, acc2A);
				acc2B = __hfma2(in2B, wB, acc2B);
			}
		}
	}

	half2* outH2 = reinterpret_cast<half2*>(outputTensor);
	int outBase1 = (outI1 * 32 + outJ1) * 768 + blockCh;
	outH2[(outBase1 + chOffsetA) / 2] = acc1A;
	outH2[(outBase1 + chOffsetB) / 2] = acc1B;

	int outBase2 = (outI2 * 32 + outJ2) * 768 + blockCh;
	outH2[(outBase2 + chOffsetA) / 2] = acc2A;
	outH2[(outBase2 + chOffsetB) / 2] = acc2B;
}

void depthwise_forward_32_768_7(half* inputTensor,
		const half* weight,
		half* outputTensor,
		int tensorHeight,
		int tensorWidth,
		int Ch,
		int windowSize,
		cudaStream_t stream) {

	if (tensorHeight != 32 || tensorWidth != 32 || Ch != 768 || windowSize != 7) {
		std::cout << "Error: depthwise_forward_32_768_7 requires 32×32×768 tensor with 7×7 kernel" << std::endl;
		exit(1);
	}

	dim3 gridSize(4, 4, 6);  // 128 channels per block
	int threadSize = 1024;
	size_t sharedMemSize = 7 * 7 * 128 * sizeof(half);

	depthwise_32_768_7_kernel<<<gridSize, threadSize, sharedMemSize, stream>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// 32×32×1536, 9×9 kernel (512-THREAD, 2 blocks/SM)
// BATCH_GROUP=2: weight loaded once, each batch in group iterates independently.
// Accumulators held in registers per-batch, shared memory tile buffer reused.
//=============================================================================
#define BATCH_GROUP_9x9 2

__global__ void __launch_bounds__(512, 2)
depthwise_32_1536_9_kernel(half* inputTensor,
                            const half* weight,
                            half* outputTensor,
                            int B,
                            int batchStride)
{
	extern __shared__ half sharedMem[];
	half* weightCache = sharedMem;
	half* inputCacheBuf[2];
	inputCacheBuf[0] = sharedMem + 9 * 9 * 64;
	inputCacheBuf[1] = sharedMem + 9 * 9 * 64 + 4096;

	int tid = threadIdx.x;
	int warpLane = tid & 31;
	int warpIdx = tid >> 5;

	int gridCh = 24;
	int batchGroupIdx = blockIdx.z / gridCh;
	int blockCh = (blockIdx.z % gridCh) * 64;

	int blockI = blockIdx.x * 8;
	int blockJ = blockIdx.y * 8;

	int warpI = __shfl_sync(0xffffffff, warpIdx / 8, 0);
	int warpJ = __shfl_sync(0xffffffff, warpIdx % 8, 0);

	const half2* weightGlobal = reinterpret_cast<const half2*>(weight);
	half2* weightCacheHalf2 = reinterpret_cast<half2*>(weightCache);

	// Load weight ONCE for all batches in group
	for (int phase = 0; phase < 5; ++phase) {
		int loadIdx = tid + phase * 512;
		int ki = loadIdx / (9 * 32);
		int rem = loadIdx % (9 * 32);
		int kj = rem / 32;
		int chIdx = rem % 32;
		int globalIdx = (ki * 9 + kj) * 1536 + blockCh + chIdx * 2;
		weightCacheHalf2[loadIdx] = weightGlobal[globalIdx / 2];
	}
	if (tid < 32) {
		int loadIdx = tid + 2560;
		int ki = loadIdx / (9 * 32);
		int rem = loadIdx % (9 * 32);
		int kj = rem / 32;
		int chIdx = rem % 32;
		int globalIdx = (ki * 9 + kj) * 1536 + blockCh + chIdx * 2;
		weightCacheHalf2[loadIdx] = weightGlobal[globalIdx / 2];
	}

	__syncthreads();

	int outI1 = blockI + warpI;
	int outI2 = blockI + warpI + 2;
	int outI3 = blockI + warpI + 4;
	int outI4 = blockI + warpI + 6;
	int outJ = blockJ + warpJ;

	int inJ_start = max(warpJ, 4 - blockJ);
	int inJ_end = min(warpJ + 9, 36 - blockJ);

	int tileCol = tid / 32;
	int tileLane = tid % 32;

	// Process batches in group
	int batchEnd = min(batchGroupIdx * BATCH_GROUP_9x9 + BATCH_GROUP_9x9, B);
	for (int bg = batchGroupIdx * BATCH_GROUP_9x9; bg < batchEnd; ++bg) {
		half2* inputBaseH2 = reinterpret_cast<half2*>(inputTensor + bg * batchStride);
		half2* outBaseH2 = reinterpret_cast<half2*>(outputTensor + bg * batchStride);

		half2 acc1 = __float2half2_rn(0.0f);
		half2 acc2 = __float2half2_rn(0.0f);
		half2 acc3 = __float2half2_rn(0.0f);
		half2 acc4 = __float2half2_rn(0.0f);

		int outOff1 = (outI1 * 32 + outJ) * 1536 + blockCh;
		int outOff2 = (outI2 * 32 + outJ) * 1536 + blockCh;
		int outOff3 = (outI3 * 32 + outJ) * 1536 + blockCh;
		int outOff4 = (outI4 * 32 + outJ) * 1536 + blockCh;

		half2* inputCacheHalf2;
		half2* nextCacheHalf2;
		int curBuf = 0;

		// Iter0
		if (blockI > 0) {
			inputCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[0]);
			for (int row = 0; row < 4; ++row) {
				int globalI = blockI - 4 + row;
				int globalJ = blockJ - 4 + tileCol;
				if (globalI >= 0 && globalJ >= 0 && globalJ < 32) {
					int offset = (globalI * 32 + globalJ) * 1536 + blockCh;
					inputCacheHalf2[(row * 16 + tileCol) * 32 + tileLane] = inputBaseH2[offset / 2 + tileLane];
				}
			}
			__syncthreads();

			nextCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[1]);
			for (int row = 0; row < 4; ++row) {
				int globalI = blockI + row;
				int globalJ = blockJ - 4 + tileCol;
				if (globalJ >= 0 && globalJ < 32) {
					int offset = (globalI * 32 + globalJ) * 1536 + blockCh;
					uint32_t shmem_addr = __cvta_generic_to_shared(&nextCacheHalf2[(row * 16 + tileCol) * 32 + tileLane]);
					const half2* src = &inputBaseH2[offset / 2 + tileLane];
					asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shmem_addr), "l"(src));
				}
			}
			asm volatile("cp.async.commit_group;\n");

			{
				int inI_start_p1 = max(warpI, 4 - blockI);
				int inI_start_p2 = max(warpI + 2, 4 - blockI);
				for (int inI = 0; inI < 4; ++inI) {
					for (int inJ = inJ_start; inJ < inJ_end; ++inJ) {
						half2 inVal = inputCacheHalf2[(inI * 16 + inJ) * 32 + warpLane];
						if (inI >= inI_start_p1) {
							int wI = inI - warpI;
							int wJ = inJ - warpJ;
							acc1 = __hfma2(inVal, weightCacheHalf2[(wI * 9 + wJ) * 32 + warpLane], acc1);
						}
						if (inI >= inI_start_p2) {
							int wI = inI - warpI - 2;
							int wJ = inJ - warpJ;
							acc2 = __hfma2(inVal, weightCacheHalf2[(wI * 9 + wJ) * 32 + warpLane], acc2);
						}
					}
				}
			}

			asm volatile("cp.async.wait_group 0;\n");
			__syncthreads();
			curBuf = 1;
		} else {
			inputCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[0]);
			for (int row = 0; row < 4; ++row) {
				int globalJ = blockJ - 4 + tileCol;
				if (globalJ >= 0 && globalJ < 32) {
					int offset = (blockI + row) * 32 * 1536 + globalJ * 1536 + blockCh;
					inputCacheHalf2[(row * 16 + tileCol) * 32 + tileLane] = inputBaseH2[offset / 2 + tileLane];
				}
			}
			__syncthreads();
			curBuf = 0;
		}

		// Iter1
		inputCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[curBuf]);
		nextCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[1 - curBuf]);

		for (int row = 0; row < 4; ++row) {
			int globalI = blockI + 4 + row;
			int globalJ = blockJ - 4 + tileCol;
			if (globalJ >= 0 && globalJ < 32) {
				int offset = (globalI * 32 + globalJ) * 1536 + blockCh;
				uint32_t shmem_addr = __cvta_generic_to_shared(&nextCacheHalf2[(row * 16 + tileCol) * 32 + tileLane]);
				const half2* src = &inputBaseH2[offset / 2 + tileLane];
				asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shmem_addr), "l"(src));
			}
		}
		asm volatile("cp.async.commit_group;\n");

		{
			for (int inI = 0; inI < 4; ++inI) {
				for (int inJ = inJ_start; inJ < inJ_end; ++inJ) {
					half2 inVal = inputCacheHalf2[(inI * 16 + inJ) * 32 + warpLane];
					int wJ = inJ - warpJ;
					acc1 = __hfma2(inVal, weightCacheHalf2[((inI - warpI + 4) * 9 + wJ) * 32 + warpLane], acc1);
					acc2 = __hfma2(inVal, weightCacheHalf2[((inI - warpI + 2) * 9 + wJ) * 32 + warpLane], acc2);
					if (inI >= warpI) {
						acc3 = __hfma2(inVal, weightCacheHalf2[((inI - warpI) * 9 + wJ) * 32 + warpLane], acc3);
					}
					if (inI >= warpI + 2) {
						acc4 = __hfma2(inVal, weightCacheHalf2[((inI - warpI - 2) * 9 + wJ) * 32 + warpLane], acc4);
					}
				}
			}
		}

		asm volatile("cp.async.wait_group 0;\n");
		__syncthreads();
		curBuf = 1 - curBuf;

		// Iter2
		inputCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[curBuf]);
		nextCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[1 - curBuf]);

		if (blockI < 24) {
			for (int row = 0; row < 4; ++row) {
				int globalI = blockI + 8 + row;
				int globalJ = blockJ - 4 + tileCol;
				if (globalI < 32 && globalJ >= 0 && globalJ < 32) {
					int offset = (globalI * 32 + globalJ) * 1536 + blockCh;
					uint32_t shmem_addr = __cvta_generic_to_shared(&nextCacheHalf2[(row * 16 + tileCol) * 32 + tileLane]);
					const half2* src = &inputBaseH2[offset / 2 + tileLane];
					asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shmem_addr), "l"(src));
				}
			}
		}
		asm volatile("cp.async.commit_group;\n");

		{
			for (int inI = 0; inI < 4; ++inI) {
				for (int inJ = inJ_start; inJ < inJ_end; ++inJ) {
					half2 inVal = inputCacheHalf2[(inI * 16 + inJ) * 32 + warpLane];
					int wJ = inJ - warpJ;
					if (inI <= warpI) {
						acc1 = __hfma2(inVal, weightCacheHalf2[((inI - warpI + 8) * 9 + wJ) * 32 + warpLane], acc1);
					}
					if (inI <= warpI + 2) {
						acc2 = __hfma2(inVal, weightCacheHalf2[((inI - warpI + 6) * 9 + wJ) * 32 + warpLane], acc2);
					}
					acc3 = __hfma2(inVal, weightCacheHalf2[((inI - warpI + 4) * 9 + wJ) * 32 + warpLane], acc3);
					acc4 = __hfma2(inVal, weightCacheHalf2[((inI - warpI + 2) * 9 + wJ) * 32 + warpLane], acc4);
				}
			}
		}

		asm volatile("cp.async.wait_group 0;\n");

		// Iter3
		if (blockI < 24) {
			__syncthreads();
			curBuf = 1 - curBuf;
			inputCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[curBuf]);

			{
				int inI_end_p3 = min(warpI + 1, 24 - blockI);
				int inI_end_p4 = min(warpI + 3, 24 - blockI);
				for (int inI = 0; inI < 4; ++inI) {
					if (inI >= inI_end_p4) break;
					for (int inJ = inJ_start; inJ < inJ_end; ++inJ) {
						half2 inVal = inputCacheHalf2[(inI * 16 + inJ) * 32 + warpLane];
						int wJ = inJ - warpJ;
						if (inI < inI_end_p3) {
							acc3 = __hfma2(inVal, weightCacheHalf2[((inI - warpI + 8) * 9 + wJ) * 32 + warpLane], acc3);
						}
						if (inI < inI_end_p4) {
							acc4 = __hfma2(inVal, weightCacheHalf2[((inI - warpI + 6) * 9 + wJ) * 32 + warpLane], acc4);
						}
					}
				}
			}
		}

		outBaseH2[outOff1 / 2 + warpLane] = acc1;
		outBaseH2[outOff2 / 2 + warpLane] = acc2;
		outBaseH2[outOff3 / 2 + warpLane] = acc3;
		outBaseH2[outOff4 / 2 + warpLane] = acc4;

		// Barrier: all warps must finish before next batch overwrites shmem tiles
		__syncthreads();
	}
}

void depthwise_forward_32_1536_9(half* inputTensor,
		const half* weight,
		half* outputTensor,
		int tensorHeight,
		int tensorWidth,
		int Ch,
		int windowSize,
		int B,
		int batchStride,
		cudaStream_t stream) {

	if (tensorHeight != 32 || tensorWidth != 32 || Ch != 1536 || windowSize != 9) {
		std::cout << "Error: depthwise_forward_32_1536_9 requires 32×32×1536 tensor with 9×9 kernel" << std::endl;
		exit(1);
	}

	int gridCh = 24;
	int numBatchGroups = (B + BATCH_GROUP_9x9 - 1) / BATCH_GROUP_9x9;
	dim3 gridSize(4, 4, gridCh * numBatchGroups);
	int threadSize = 512;
	size_t sharedMemSize = (9 * 9 * 64 + 2 * 4 * 16 * 32 * 2) * sizeof(half);

	depthwise_32_1536_9_kernel<<<gridSize, threadSize, sharedMemSize, stream>>>(inputTensor, weight, outputTensor, B, batchStride);
}


//=============================================================================
// 16×16×1536, 9×9 kernel (512-THREAD, 2 blocks/SM)
// Using the 512-thread variant from depthWise_ch_1536_k_9_t_16_half_forward_512.cu
//=============================================================================
__global__ void __launch_bounds__(512, 2)
depthwise_16_1536_9_kernel(half* inputTensor,
                            const half* weight,
                            half* outputTensor)
{
	// Identical implementation to depthwise_32_1536_9_kernel but with:
	// - Spatial size 16 instead of 32
	// - Grid 2×2×24 instead of 4×4×24
	// - Boundary checks adjusted for 16×16

	extern __shared__ half sharedMem[];
	half* weightCache = sharedMem;
	half* inputCacheBuf[2];
	inputCacheBuf[0] = sharedMem + 9 * 9 * 64;
	inputCacheBuf[1] = sharedMem + 9 * 9 * 64 + 4096;

	int tid = threadIdx.x;
	int warpLane = tid & 31;
	int warpIdx = tid >> 5;

	int blockI = blockIdx.x * 8;
	int blockJ = blockIdx.y * 8;
	int blockCh = blockIdx.z * 64;

	int warpI = __shfl_sync(0xffffffff, warpIdx / 8, 0);
	int warpJ = __shfl_sync(0xffffffff, warpIdx % 8, 0);

	const half2* weightGlobal = reinterpret_cast<const half2*>(weight);
	half2* weightCacheHalf2 = reinterpret_cast<half2*>(weightCache);
	half2* inputTensorHalf2 = reinterpret_cast<half2*>(inputTensor);
	half2* outHalf2 = reinterpret_cast<half2*>(outputTensor);

	// Weight load (identical to 32×32 version)
	for (int phase = 0; phase < 5; ++phase) {
		int loadIdx = tid + phase * 512;
		int ki = loadIdx / (9 * 32);
		int rem = loadIdx % (9 * 32);
		int kj = rem / 32;
		int chIdx = rem % 32;
		int globalIdx = (ki * 9 + kj) * 1536 + blockCh + chIdx * 2;
		weightCacheHalf2[loadIdx] = weightGlobal[globalIdx / 2];
	}
	if (tid < 32) {
		int loadIdx = tid + 2560;
		int ki = loadIdx / (9 * 32);
		int rem = loadIdx % (9 * 32);
		int kj = rem / 32;
		int chIdx = rem % 32;
		int globalIdx = (ki * 9 + kj) * 1536 + blockCh + chIdx * 2;
		weightCacheHalf2[loadIdx] = weightGlobal[globalIdx / 2];
	}

	half2 acc1 = __float2half2_rn(0.0f);
	half2 acc2 = __float2half2_rn(0.0f);
	half2 acc3 = __float2half2_rn(0.0f);
	half2 acc4 = __float2half2_rn(0.0f);

	__syncthreads();

	int outI1 = blockI + warpI;
	int outI2 = blockI + warpI + 2;
	int outI3 = blockI + warpI + 4;
	int outI4 = blockI + warpI + 6;
	int outJ = blockJ + warpJ;

	int outOffset1 = (outI1 * 16 + outJ) * 1536 + blockCh;
	int outOffset2 = (outI2 * 16 + outJ) * 1536 + blockCh;
	int outOffset3 = (outI3 * 16 + outJ) * 1536 + blockCh;
	int outOffset4 = (outI4 * 16 + outJ) * 1536 + blockCh;

	int inJ_start = max(warpJ, 4 - blockJ);
	int inJ_end = min(warpJ + 9, 20 - blockJ);

	int tileCol = tid / 32;
	int tileLane = tid % 32;

	int curBuf = 0;
	half2* inputCacheHalf2;
	half2* nextCacheHalf2;

	// Iter0
	if (blockI > 0) {
		inputCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[0]);
		for (int row = 0; row < 4; ++row) {
			int globalI = blockI - 4 + row;
			int globalJ = blockJ - 4 + tileCol;
			if (globalI >= 0 && globalJ >= 0 && globalJ < 16) {
				int offset = (globalI * 16 + globalJ) * 1536 + blockCh;
				inputCacheHalf2[(row * 16 + tileCol) * 32 + tileLane] = inputTensorHalf2[offset / 2 + tileLane];
			}
		}
		__syncthreads();

		nextCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[1]);
		for (int row = 0; row < 4; ++row) {
			int globalI = blockI + row;
			int globalJ = blockJ - 4 + tileCol;
			if (globalJ >= 0 && globalJ < 16) {
				int offset = (globalI * 16 + globalJ) * 1536 + blockCh;
				uint32_t shmem_addr = __cvta_generic_to_shared(&nextCacheHalf2[(row * 16 + tileCol) * 32 + tileLane]);
				const half2* src = &inputTensorHalf2[offset / 2 + tileLane];
				asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shmem_addr), "l"(src));
			}
		}
		asm volatile("cp.async.commit_group;\n");

		{
			int inI_start_p1 = max(warpI, 4 - blockI);
			int inI_start_p2 = max(warpI + 2, 4 - blockI);
			for (int inI = 0; inI < 4; ++inI) {
				for (int inJ = inJ_start; inJ < inJ_end; ++inJ) {
					half2 inputVal = inputCacheHalf2[(inI * 16 + inJ) * 32 + warpLane];
					if (inI >= inI_start_p1) {
						int wI = inI - warpI;
						int wJ = inJ - warpJ;
						acc1 = __hfma2(inputVal, weightCacheHalf2[(wI * 9 + wJ) * 32 + warpLane], acc1);
					}
					if (inI >= inI_start_p2) {
						int wI = inI - warpI - 2;
						int wJ = inJ - warpJ;
						acc2 = __hfma2(inputVal, weightCacheHalf2[(wI * 9 + wJ) * 32 + warpLane], acc2);
					}
				}
			}
		}

		asm volatile("cp.async.wait_group 0;\n");
		__syncthreads();
		curBuf = 1;
	} else {
		inputCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[0]);
		for (int row = 0; row < 4; ++row) {
			int globalJ = blockJ - 4 + tileCol;
			if (globalJ >= 0 && globalJ < 16) {
				int offset = (blockI + row) * 16 * 1536 + globalJ * 1536 + blockCh;
				inputCacheHalf2[(row * 16 + tileCol) * 32 + tileLane] = inputTensorHalf2[offset / 2 + tileLane];
			}
		}
		__syncthreads();
		curBuf = 0;
	}

	// Iter1
	inputCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[curBuf]);
	nextCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[1 - curBuf]);

	for (int row = 0; row < 4; ++row) {
		int globalI = blockI + 4 + row;
		int globalJ = blockJ - 4 + tileCol;
		if (globalI < 16 && globalJ >= 0 && globalJ < 16) {
			int offset = (globalI * 16 + globalJ) * 1536 + blockCh;
			uint32_t shmem_addr = __cvta_generic_to_shared(&nextCacheHalf2[(row * 16 + tileCol) * 32 + tileLane]);
			const half2* src = &inputTensorHalf2[offset / 2 + tileLane];
			asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shmem_addr), "l"(src));
		}
	}
	asm volatile("cp.async.commit_group;\n");

	{
		for (int inI = 0; inI < 4; ++inI) {
			for (int inJ = inJ_start; inJ < inJ_end; ++inJ) {
				half2 inputVal = inputCacheHalf2[(inI * 16 + inJ) * 32 + warpLane];
				int wJ = inJ - warpJ;
				acc1 = __hfma2(inputVal, weightCacheHalf2[((inI - warpI + 4) * 9 + wJ) * 32 + warpLane], acc1);
				acc2 = __hfma2(inputVal, weightCacheHalf2[((inI - warpI + 2) * 9 + wJ) * 32 + warpLane], acc2);
				if (inI >= warpI) {
					acc3 = __hfma2(inputVal, weightCacheHalf2[((inI - warpI) * 9 + wJ) * 32 + warpLane], acc3);
				}
				if (inI >= warpI + 2) {
					acc4 = __hfma2(inputVal, weightCacheHalf2[((inI - warpI - 2) * 9 + wJ) * 32 + warpLane], acc4);
				}
			}
		}
	}

	asm volatile("cp.async.wait_group 0;\n");
	__syncthreads();
	curBuf = 1 - curBuf;

	// Iter2
	inputCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[curBuf]);
	nextCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[1 - curBuf]);

	if (blockI < 8) {
		for (int row = 0; row < 4; ++row) {
			int globalI = blockI + 8 + row;
			int globalJ = blockJ - 4 + tileCol;
			if (globalI < 16 && globalJ >= 0 && globalJ < 16) {
				int offset = (globalI * 16 + globalJ) * 1536 + blockCh;
				uint32_t shmem_addr = __cvta_generic_to_shared(&nextCacheHalf2[(row * 16 + tileCol) * 32 + tileLane]);
				const half2* src = &inputTensorHalf2[offset / 2 + tileLane];
				asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shmem_addr), "l"(src));
			}
		}
	}
	asm volatile("cp.async.commit_group;\n");

	{
		for (int inI = 0; inI < 4; ++inI) {
			for (int inJ = inJ_start; inJ < inJ_end; ++inJ) {
				half2 inputVal = inputCacheHalf2[(inI * 16 + inJ) * 32 + warpLane];
				int wJ = inJ - warpJ;
				if (inI <= warpI) {
					acc1 = __hfma2(inputVal, weightCacheHalf2[((inI - warpI + 8) * 9 + wJ) * 32 + warpLane], acc1);
				}
				if (inI <= warpI + 2) {
					acc2 = __hfma2(inputVal, weightCacheHalf2[((inI - warpI + 6) * 9 + wJ) * 32 + warpLane], acc2);
				}
				acc3 = __hfma2(inputVal, weightCacheHalf2[((inI - warpI + 4) * 9 + wJ) * 32 + warpLane], acc3);
				acc4 = __hfma2(inputVal, weightCacheHalf2[((inI - warpI + 2) * 9 + wJ) * 32 + warpLane], acc4);
			}
		}
	}

	asm volatile("cp.async.wait_group 0;\n");

	// Iter3
	if (blockI < 8) {
		__syncthreads();
		curBuf = 1 - curBuf;
		inputCacheHalf2 = reinterpret_cast<half2*>(inputCacheBuf[curBuf]);

		{
			int inI_end_p3 = min(warpI + 1, 8 - blockI);
			int inI_end_p4 = min(warpI + 3, 8 - blockI);
			for (int inI = 0; inI < 4; ++inI) {
				if (inI >= inI_end_p4) break;
				for (int inJ = inJ_start; inJ < inJ_end; ++inJ) {
					half2 inputVal = inputCacheHalf2[(inI * 16 + inJ) * 32 + warpLane];
					int wJ = inJ - warpJ;
					if (inI < inI_end_p3) {
						acc3 = __hfma2(inputVal, weightCacheHalf2[((inI - warpI + 8) * 9 + wJ) * 32 + warpLane], acc3);
					}
					if (inI < inI_end_p4) {
						acc4 = __hfma2(inputVal, weightCacheHalf2[((inI - warpI + 6) * 9 + wJ) * 32 + warpLane], acc4);
					}
				}
			}
		}
	}

	outHalf2[outOffset1 / 2 + warpLane] = acc1;
	outHalf2[outOffset2 / 2 + warpLane] = acc2;
	outHalf2[outOffset3 / 2 + warpLane] = acc3;
	outHalf2[outOffset4 / 2 + warpLane] = acc4;
}

void depthwise_forward_16_1536_9(half* inputTensor,
		const half* weight,
		half* outputTensor,
		int tensorHeight,
		int tensorWidth,
		int Ch,
		int windowSize,
		cudaStream_t stream) {

	if (tensorHeight != 16 || tensorWidth != 16 || Ch != 1536 || windowSize != 9) {
		std::cout << "Error: depthwise_forward_16_1536_9 requires 16×16×1536 tensor with 9×9 kernel" << std::endl;
		exit(1);
	}

	dim3 gridSize(2, 2, 24);
	int threadSize = 512;
	size_t sharedMemSize = (9 * 9 * 64 + 2 * 4 * 16 * 32 * 2) * sizeof(half);

	depthwise_16_1536_9_kernel<<<gridSize, threadSize, sharedMemSize, stream>>>(inputTensor, weight, outputTensor);
}


//=============================================================================
// 32×32×1536, 13×13 kernel (1024-THREAD, shared memory)
// Using single-load variant from depthWise_ch_1536_k_13_t_32_half_forward.cu
//=============================================================================
// NOTE: This is a placeholder - full 13×13 kernel implementation would go here
// For now, using error message to indicate it needs to be copied from the
// 13×13 kernel file (depthWise_ch_1536_k_13_t_32_half_forward.cu)
void depthwise_forward_32_1536_13(half* inputTensor,
		const half* weight,
		half* outputTensor,
		int tensorHeight,
		int tensorWidth,
		int Ch,
		int windowSize) {

	if (tensorHeight != 32 || tensorWidth != 32 || Ch != 1536 || windowSize != 13) {
		std::cout << "Error: depthwise_forward_32_1536_13 requires 32×32×1536 tensor with 13×13 kernel" << std::endl;
		exit(1);
	}

	// TODO: Implement or copy from depthWise_ch_1536_k_13_t_32_half_forward.cu
	std::cout << "Error: depthwise_forward_32_1536_13 not yet implemented in integration file" << std::endl;
	std::cout << "Please copy the single-load kernel from depthWise_ch_1536_k_13_t_32_half_forward.cu" << std::endl;
	exit(1);
}


//=============================================================================
// Python bindings using PyBind11
//=============================================================================
#include <torch/extension.h>

// Wrapper functions that accept PyTorch tensors
void depthwise_forward_32_768_7_torch(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize) {

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();
    depthwise_forward_32_768_7(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        tensorHeight, tensorWidth, Ch, windowSize, stream
    );
}

void depthwise_forward_32_1536_9_torch(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize) {

    int B = input.size(0);
    int batchStride = tensorHeight * tensorWidth * Ch;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();

    depthwise_forward_32_1536_9(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        tensorHeight, tensorWidth, Ch, windowSize,
        B, batchStride, stream
    );
}

void depthwise_forward_16_1536_9_torch(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize) {

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();
    depthwise_forward_16_1536_9(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        tensorHeight, tensorWidth, Ch, windowSize, stream
    );
}

void depthwise_forward_32_1536_13_torch(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor output,
    int tensorHeight,
    int tensorWidth,
    int Ch,
    int windowSize) {

    depthwise_forward_32_1536_13(
        reinterpret_cast<half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        tensorHeight, tensorWidth, Ch, windowSize
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_forward_32_768_7", &depthwise_forward_32_768_7_torch, "Depthwise conv 32x32x768, 7x7 kernel");
    m.def("depthwise_forward_32_1536_9", &depthwise_forward_32_1536_9_torch, "Depthwise conv 32x32x1536, 9x9 kernel");
    m.def("depthwise_forward_16_1536_9", &depthwise_forward_16_1536_9_torch, "Depthwise conv 16x16x1536, 9x9 kernel");
    m.def("depthwise_forward_32_1536_13", &depthwise_forward_32_1536_13_torch, "Depthwise conv 32x32x1536, 13x13 kernel");
}
