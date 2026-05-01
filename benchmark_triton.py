#!/usr/bin/env python3
"""
Triton depthwise convolution baseline.

Fair comparison: FP16, NHWC layout, same padding, auto-tuned tile sizes.
Tests kernel-only latency for each configuration used in the paper.

Usage:
    python benchmark_triton.py --device cuda:0 --num_runs 500
"""

import torch
import triton
import triton.language as tl
import argparse
import csv
import os


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=8, num_stages=2),
    ],
    key=["H", "W", "C", "K"],
)
@triton.jit
def depthwise_conv2d_nhwc_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    H,
    W,
    C,
    K,
    PAD,
    stride_h,
    stride_w,
    BLOCK_C: tl.constexpr,
):
    """
    Depthwise conv2d on NHWC tensor, batch=1.
    Each program handles one output pixel for BLOCK_C channels.

    Input:  [H, W, C]
    Weight: [K, K, C]
    Output: [H, W, C]
    """
    pid = tl.program_id(0)
    num_hw = H * W
    num_c_blocks = tl.cdiv(C, BLOCK_C)

    hw_idx = pid // num_c_blocks
    c_block = pid % num_c_blocks

    oh = hw_idx // W
    ow = hw_idx % W
    c_start = c_block * BLOCK_C

    c_offsets = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for kh in range(K):
        ih = oh + kh - PAD
        if ih >= 0 and ih < H:
            for kw in range(K):
                iw = ow + kw - PAD
                if iw >= 0 and iw < W:
                    in_off = ih * stride_h + iw * stride_w + c_offsets
                    x_vals = tl.load(input_ptr + in_off, mask=c_mask, other=0.0).to(tl.float32)
                    w_off = kh * K * C + kw * C + c_offsets
                    w_vals = tl.load(weight_ptr + w_off, mask=c_mask, other=0.0).to(tl.float32)
                    acc += x_vals * w_vals

    out_off = oh * stride_h + ow * stride_w + c_offsets
    tl.store(output_ptr + out_off, acc.to(tl.float16), mask=c_mask)


def triton_depthwise_conv2d(x, weight, K, pad):
    """
    x:      [H, W, C] float16
    weight: [K, K, C] float16
    returns: [H, W, C] float16
    """
    H, W, C = x.shape
    output = torch.empty_like(x)

    stride_h = W * C
    stride_w = C

    grid = lambda meta: (H * W * triton.cdiv(C, meta["BLOCK_C"]),)

    depthwise_conv2d_nhwc_kernel[grid](
        x, weight, output,
        H, W, C, K, pad,
        stride_h, stride_w,
    )
    return output


def benchmark_triton_kernel(C, K, H, W, device, num_runs=500, warmup=200):
    """Benchmark Triton depthwise conv for one configuration."""
    pad = K // 2
    x = torch.randn(H, W, C, device=device, dtype=torch.float16)
    weight = torch.randn(K, K, C, device=device, dtype=torch.float16)

    for _ in range(warmup):
        out = triton_depthwise_conv2d(x, weight, K, pad)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_runs):
        out = triton_depthwise_conv2d(x, weight, K, pad)
    end.record()
    torch.cuda.synchronize(device)

    latency_ms = start.elapsed_time(end) / num_runs

    del x, weight, out
    torch.cuda.empty_cache()
    return latency_ms


def verify_correctness(C, K, H, W, device):
    """Verify Triton kernel matches PyTorch reference."""
    pad = K // 2
    x_nhwc = torch.randn(H, W, C, device=device, dtype=torch.float16)
    w_nhwc = torch.randn(K, K, C, device=device, dtype=torch.float16)

    out_triton = triton_depthwise_conv2d(x_nhwc, w_nhwc, K, pad)

    x_nchw = x_nhwc.permute(2, 0, 1).unsqueeze(0)
    w_nchw = w_nhwc.permute(2, 0, 1).unsqueeze(1)
    ref = torch.nn.functional.conv2d(x_nchw.float(), w_nchw.float(), padding=pad, groups=C)
    ref_nhwc = ref.squeeze(0).permute(1, 2, 0).half()

    max_diff = (out_triton.float() - ref_nhwc.float()).abs().max().item()
    return max_diff


CONFIGS = [
    ("ConvMixer 9x9/1536ch/32x32",  1536, 9,  32, 32),
    ("ConvMixer 7x7/768ch/32x32",    768, 7,  32, 32),
    ("RepLKNet 31x31/128ch/56x56",   128, 31, 56, 56),
    ("RepLKNet 29x29/256ch/28x28",   256, 29, 28, 28),
    ("RepLKNet 27x27/512ch/14x14",   512, 27, 14, 14),
    ("RepLKNet 13x13/1024ch/7x7",   1024, 13,  7,  7),
]


def main():
    parser = argparse.ArgumentParser(description="Triton depthwise conv baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_runs", type=int, default=500)
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "spark_triton_results.csv")

    print("=" * 78)
    print("Triton Depthwise Conv2d Baseline (batch=1, FP16, NHWC)")
    print("=" * 78)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}")
    print(f"Runs per config: {args.num_runs}")
    print()

    if not args.skip_verify:
        print("Verifying correctness...")
        for name, C, K, H, W in CONFIGS:
            max_diff = verify_correctness(C, K, H, W, device)
            status = "PASS" if max_diff < 1.0 else "FAIL"
            print(f"  {name}: max_diff={max_diff:.4f} [{status}]")
        print()

    header = f"{'Config':<32} {'Triton (ms)':>12}"
    print(header)
    print("-" * 50)

    rows = []
    for name, C, K, H, W in CONFIGS:
        latency = benchmark_triton_kernel(C, K, H, W, device, args.num_runs)
        print(f"{name:<32} {latency:>12.3f}")
        rows.append({
            "config": name,
            "channels": C,
            "kernel_size": K,
            "spatial": f"{H}x{W}",
            "triton_ms": f"{latency:.3f}",
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
