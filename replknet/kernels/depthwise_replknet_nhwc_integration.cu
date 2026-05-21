/*
 * RepLKNet Depthwise Convolution — PyTorch NHWC Integration
 *
 * Launches the custom __global__ kernels on PyTorch's current CUDA stream.
 * Accepts channels_last [B,C,H,W] tensors directly (physical layout = NHWC).
 * Adds bias after convolution via a fused kernel on the same stream.
 *
 * Kernel .cu files (compiled separately, linked by JIT):
 *   depthWise_ch_128_k_31_t_56_half_forward.cu
 *   depthWise_ch_256_k_29_t_28_half_forward.cu
 *   depthWise_ch_512_k_27_t_14_half_forward.cu
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "depthwise_replknet_nhwc_integration.hpp"

// ---- Bias-add kernel (launched on same stream) ----

__global__ void bias_add_nhwc_kernel(half* output, const half* bias, int HWC, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < HWC) {
        output[idx] = __hadd(output[idx], bias[idx % C]);
    }
}

// ---- Launch configs (from the standalone C_interface functions) ----

// 31×31, 128ch, 56×56: grid(7,5,4)=140 blocks, 512 threads, shmem=31*31*32*2
static constexpr dim3 GRID_128_31 = {7, 5, 4};
static constexpr int THREADS_128_31 = 512;
static constexpr size_t SHMEM_128_31 = 31 * 31 * 32 * sizeof(half);

// 29×29, 256ch, 28×28: grid(4,3,8)=96 blocks, 512 threads, shmem=29*29*32*2
static constexpr dim3 GRID_256_29 = {4, 3, 8};
static constexpr int THREADS_256_29 = 512;
static constexpr size_t SHMEM_256_29 = 29 * 29 * 32 * sizeof(half);

// 27×27, 512ch, 14×14 — allshared branchfree h2 cb32: grid(2,7,16)=224, 256 threads
static constexpr dim3 GRID_512_27 = {2, 7, 16};
static constexpr int THREADS_512_27 = 256;
static constexpr size_t SHMEM_512_27 = (27 * 27 * 32 + 14 * 14 * 32) * sizeof(half);

// ---- Helper: launch bias add on stream ----

static void launch_bias_add(half* output, const half* bias, int H, int W, int C,
                            cudaStream_t stream) {
    int HWC = H * W * C;
    int threads = 256;
    int blocks = (HWC + threads - 1) / threads;
    bias_add_nhwc_kernel<<<blocks, threads, 0, stream>>>(output, bias, HWC, C);
}

// ====================================================================
// 31×31, 128ch, 56×56
// ====================================================================
torch::Tensor depthwise_31x31_128_nhwc(
    torch::Tensor input,   // [B, 128, 56, 56] channels_last FP16
    torch::Tensor weight,  // [31, 31, 128] contiguous FP16
    torch::Tensor bias)    // [128] FP16, or empty tensor
{
    TORCH_CHECK(input.is_contiguous(at::MemoryFormat::ChannelsLast),
                "input must be channels_last");
    TORCH_CHECK(weight.is_contiguous() && weight.dtype() == torch::kFloat16);

    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    TORCH_CHECK(C == 128 && H == 56 && W == 56);

    auto output = torch::empty_like(input);
    bool has_bias = bias.numel() > 0;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();

    cudaFuncSetAttribute(
        depthWise_forward_ch_128_kernel_31_tensor_56_fp16_colmerge_8x12,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_128_31);

    int batch_stride = H * W * C;
    for (int b = 0; b < B; b++) {
        half* in_ptr  = reinterpret_cast<half*>(input.data_ptr<at::Half>()) + b * batch_stride;
        half* out_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>()) + b * batch_stride;

        depthWise_forward_ch_128_kernel_31_tensor_56_fp16_colmerge_8x12
            <<<GRID_128_31, THREADS_128_31, SHMEM_128_31, stream>>>(
                in_ptr,
                reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
                out_ptr);

        if (has_bias)
            launch_bias_add(out_ptr,
                           reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
                           H, W, C, stream);
    }
    return output;
}

// ====================================================================
// 29×29, 256ch, 28×28
// ====================================================================
torch::Tensor depthwise_29x29_256_nhwc(
    torch::Tensor input,   // [B, 256, 28, 28] channels_last FP16
    torch::Tensor weight,  // [29, 29, 256] contiguous FP16
    torch::Tensor bias)    // [256] FP16, or empty tensor
{
    TORCH_CHECK(input.is_contiguous(at::MemoryFormat::ChannelsLast),
                "input must be channels_last");
    TORCH_CHECK(weight.is_contiguous() && weight.dtype() == torch::kFloat16);

    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    TORCH_CHECK(C == 256 && H == 28 && W == 28);

    auto output = torch::empty_like(input);
    bool has_bias = bias.numel() > 0;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();

    cudaFuncSetAttribute(
        depthWise_forward_ch_256_kernel_29_tensor_28_fp16_colmerge_8x12,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_256_29);

    int batch_stride = H * W * C;
    for (int b = 0; b < B; b++) {
        half* in_ptr  = reinterpret_cast<half*>(input.data_ptr<at::Half>()) + b * batch_stride;
        half* out_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>()) + b * batch_stride;

        depthWise_forward_ch_256_kernel_29_tensor_28_fp16_colmerge_8x12
            <<<GRID_256_29, THREADS_256_29, SHMEM_256_29, stream>>>(
                in_ptr,
                reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
                out_ptr);

        if (has_bias)
            launch_bias_add(out_ptr,
                           reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
                           H, W, C, stream);
    }
    return output;
}

// ====================================================================
// 27×27, 512ch, 14×14
// ====================================================================
torch::Tensor depthwise_27x27_512_nhwc(
    torch::Tensor input,   // [B, 512, 14, 14] channels_last FP16
    torch::Tensor weight,  // [27, 27, 512] contiguous FP16
    torch::Tensor bias)    // [512] FP16, or empty tensor
{
    TORCH_CHECK(input.is_contiguous(at::MemoryFormat::ChannelsLast),
                "input must be channels_last");
    TORCH_CHECK(weight.is_contiguous() && weight.dtype() == torch::kFloat16);

    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    TORCH_CHECK(C == 512 && H == 14 && W == 14);

    auto output = torch::empty_like(input);
    bool has_bias = bias.numel() > 0;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();

    cudaFuncSetAttribute(
        depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_branchfree_h2_cb32,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_512_27);

    int batch_stride = H * W * C;
    for (int b = 0; b < B; b++) {
        half* in_ptr  = reinterpret_cast<half*>(input.data_ptr<at::Half>()) + b * batch_stride;
        half* out_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>()) + b * batch_stride;

        depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_branchfree_h2_cb32
            <<<GRID_512_27, THREADS_512_27, SHMEM_512_27, stream>>>(
                in_ptr,
                reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
                out_ptr);

        if (has_bias)
            launch_bias_add(out_ptr,
                           reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
                           H, W, C, stream);
    }
    return output;
}

// ====================================================================
// PyBind11 Module
// ====================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_31x31_128_nhwc", &depthwise_31x31_128_nhwc,
          "31x31 depthwise, 128ch 56x56, NHWC channels_last (colmerge 8x12)");
    m.def("depthwise_29x29_256_nhwc", &depthwise_29x29_256_nhwc,
          "29x29 depthwise, 256ch 28x28, NHWC channels_last (colmerge 8x12)");
    m.def("depthwise_27x27_512_nhwc", &depthwise_27x27_512_nhwc,
          "27x27 depthwise, 512ch 14x14, NHWC channels_last (naive64 half2)");
}
