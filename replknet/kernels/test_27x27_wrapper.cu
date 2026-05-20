/*
 * Test wrapper for 27×27/512ch/14×14 kernel variants.
 * JIT-compiled by test_27x27_variants.py.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "depthWise512_14_k27.hpp"

static torch::Tensor run_variant(
    torch::Tensor input,
    torch::Tensor weight,
    void (*c_fn)(half*, const half*, half*, int, int, int, int))
{
    TORCH_CHECK(input.is_contiguous(at::MemoryFormat::ChannelsLast));
    TORCH_CHECK(weight.is_contiguous() && weight.dtype() == torch::kFloat16);

    int B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    auto output = torch::empty_like(input);

    int batch_stride = H * W * C;
    for (int b = 0; b < B; b++) {
        half* in_ptr  = reinterpret_cast<half*>(input.data_ptr<at::Half>()) + b * batch_stride;
        half* out_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>()) + b * batch_stride;
        c_fn(in_ptr,
             reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
             out_ptr, H, W, C, 27);
    }
    return output;
}

torch::Tensor test_naive(torch::Tensor input, torch::Tensor weight) {
    return run_variant(input, weight,
        depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive_C_interface);
}

torch::Tensor test_allshared(torch::Tensor input, torch::Tensor weight) {
    return run_variant(input, weight,
        depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_C_interface);
}

torch::Tensor test_naive64(torch::Tensor input, torch::Tensor weight) {
    return run_variant(input, weight,
        depthWise_forward_ch_512_kernel_27_tensor_14_fp16_naive64_C_interface);
}

torch::Tensor test_allshared_h2_cb32(torch::Tensor input, torch::Tensor weight) {
    return run_variant(input, weight,
        depthWise_forward_ch_512_kernel_27_tensor_14_fp16_allshared_branchfree_h2_cb32_C_interface);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_naive", &test_naive);
    m.def("test_allshared", &test_allshared);
    m.def("test_naive64", &test_naive64);
    m.def("test_allshared_h2_cb32", &test_allshared_h2_cb32);
}
