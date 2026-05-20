"""
Test and benchmark 27×27/512ch/14×14 kernel variants.
Verifies correctness against PyTorch's depthwise conv, then benchmarks.

Usage:
    python test_27x27_variants.py [--num_runs 500] [--device cuda:0]
"""

import torch
import torch.nn.functional as F
import os
import argparse
from torch.utils.cpp_extension import load

def build_extension():
    kernel_dir = os.path.dirname(os.path.abspath(__file__))
    return load(
        name="depthwise_27x27_test",
        sources=[
            os.path.join(kernel_dir, "test_27x27_wrapper.cu"),
            os.path.join(kernel_dir, "depthWise_ch_512_k_27_t_14_half_forward.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        extra_include_paths=[kernel_dir],
        verbose=True
    )

def reference_conv(input_tensor, weight_khwc, padding=13):
    C = weight_khwc.shape[2]
    K = weight_khwc.shape[0]
    w = weight_khwc.permute(2, 0, 1).unsqueeze(1).contiguous()
    return F.conv2d(input_tensor.float(), w.float(), padding=padding, groups=C).half()

def benchmark_kernel(fn, num_runs=500, warmup=100):
    flush = torch.empty(32 * 1024 * 1024, dtype=torch.float32, device='cuda')
    times = []
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(num_runs):
        flush.fill_(1.0)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    trim = num_runs // 10
    trimmed = times[trim:-trim] if trim > 0 else times
    return sum(trimmed) / len(trimmed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = args.device

    B, C, H, W, K = 1, 512, 14, 14, 27

    print("Building CUDA extension...")
    ext = build_extension()
    print("Build complete.\n")

    x = torch.randn(B, C, H, W, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    w_khwc = torch.randn(K, K, C, device=device, dtype=torch.float16).contiguous()

    ref = reference_conv(x, w_khwc, padding=13).to(memory_format=torch.channels_last)

    variants = [
        ("naive (Cb=32, half, global input)", ext.test_naive),
        ("allshared (Cb=32, half, colmerge)", ext.test_allshared),
        ("naive64 (Cb=64, half2, global input)", ext.test_naive64),
        ("allshared_h2_cb32 (Cb=32, half2, split-warp)", ext.test_allshared_h2_cb32),
    ]

    print(f"Config: {K}x{K} depthwise, {C}ch, {H}x{W}, FP16, B={B}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Benchmark: {args.num_runs} runs, cold-L2, trimmed mean\n")

    print(f"{'Variant':<50} {'Correct':>16} {'Latency (us)':>14} {'vs naive64':>10}")
    print("-" * 95)

    baseline_us = None

    for name, fn in variants:
        out = fn(x, w_khwc)
        max_err = (out.float() - ref.float()).abs().max().item()
        correct = max_err < 0.5

        lat_ms = benchmark_kernel(lambda: fn(x, w_khwc), num_runs=args.num_runs)
        lat_us = lat_ms * 1000

        if "naive64" in name:
            baseline_us = lat_us

        speedup = ""
        if baseline_us is not None and baseline_us > 0:
            speedup = f"{baseline_us / lat_us:.2f}x"

        status = f"ok (err={max_err:.4f})" if correct else f"FAIL (err={max_err:.4f})"
        print(f"{name:<50} {status:>16} {lat_us:>11.1f} us {speedup:>10}")

    print()

if __name__ == "__main__":
    main()
