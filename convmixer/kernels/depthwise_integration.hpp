
#ifndef DEPTHWISE_INTEGRATION_H
#define DEPTHWISE_INTEGRATION_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_fp16.h>

//=============================================================================
// Integration-ready depthwise convolution kernels for PyTorch
// Naming convention: depthwise_forward_tensorSize_channel_windowSize
// Format: tensorSize = spatial dimension, channel count, kernel window size
//
// All kernels use FP16 (half precision) and NHWC (channels-last) layout
// All kernels use shared memory for weight caching and input tiling
// (except 32_768_7 which uses naive implementation for optimal performance)
//=============================================================================

// 32×32×768, 7×7 kernel (naive implementation - best for this config)
// Grid: 4×4×6 = 96 blocks (128ch/block for better SM utilization)
// Threads: 1024/block
// Shared memory: 12.5KB (weight cache only)
void depthwise_forward_32_768_7(half* inputTensor,
		const half* weight,
		half* outputTensor,
		int tensorHeight,
		int tensorWidth,
		int Ch,
		int windowSize);

// 32×32×1536, 9×9 kernel (512-thread, 2 blocks/SM, shared memory)
// Grid: 4×4×24 = 384 blocks (64ch/block)
// Threads: 512/block
// Shared memory: 26.1KB (weight + 2× input double-buffer)
// Best variant: barrier stall hiding via 2 blocks/SM occupancy
void depthwise_forward_32_1536_9(half* inputTensor,
		const half* weight,
		half* outputTensor,
		int tensorHeight,
		int tensorWidth,
		int Ch,
		int windowSize);

// 16×16×1536, 9×9 kernel (512-thread, 2 blocks/SM, shared memory)
// Grid: 2×2×24 = 96 blocks (64ch/block)
// Threads: 512/block
// Shared memory: 26.1KB (weight + 2× input double-buffer)
// Best variant: 1.13× faster than naive via 2 blocks/SM occupancy
void depthwise_forward_16_1536_9(half* inputTensor,
		const half* weight,
		half* outputTensor,
		int tensorHeight,
		int tensorWidth,
		int Ch,
		int windowSize);

// 32×32×1536, 13×13 kernel (1024-thread, shared memory)
// Grid: 4×4×24 = 384 blocks (64ch/block)
// Threads: 1024/block
// Shared memory: 42.1KB (weight + 2× input double-buffer)
// Uses 5 iterations with 4×20 tiles
void depthwise_forward_32_1536_13(half* inputTensor,
		const half* weight,
		half* outputTensor,
		int tensorHeight,
		int tensorWidth,
		int Ch,
		int windowSize);

#endif
