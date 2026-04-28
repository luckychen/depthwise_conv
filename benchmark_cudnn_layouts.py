#!/usr/bin/env python3
"""
cuDNN best-layout baseline: benchmark depthwise conv with both NCHW and NHWC.

Tests nn.Conv2d(groups=C) with cudnn.benchmark=True for each layout,
reporting cuDNN's best achievable latency per configuration.

Usage:
    python benchmark_cudnn_layouts.py --device cuda:0 --num_runs 500
"""

import torch
import torch.nn as nn
import argparse
import csv
import os

CONFIGS = [
    # (name, channels, kernel_size, spatial_h, spatial_w)
    ("ConvMixer 9x9/1536ch/32x32",  1536, 9,  32, 32),
    ("ConvMixer 7x7/768ch/32x32",    768, 7,  32, 32),
    ("RepLKNet 31x31/128ch/56x56",   128, 31, 56, 56),
    ("RepLKNet 29x29/256ch/28x28",   256, 29, 28, 28),
    ("RepLKNet 27x27/512ch/14x14",   512, 27, 14, 14),
    ("RepLKNet 13x13/1024ch/7x7",   1024, 13,  7,  7),
]


def benchmark_kernel(channels, kernel_size, H, W, layout, device, num_runs=500, warmup=200):
    """Benchmark a single depthwise conv config with a given layout."""
    conv = nn.Conv2d(channels, channels, kernel_size, groups=channels,
                     padding=kernel_size // 2, bias=False).to(device).half()

    if layout == "NHWC":
        conv = conv.to(memory_format=torch.channels_last)
        x = torch.randn(1, channels, H, W, device=device, dtype=torch.float16).to(
            memory_format=torch.channels_last)
    else:
        x = torch.randn(1, channels, H, W, device=device, dtype=torch.float16).contiguous()

    conv.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = conv(x)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_runs):
            _ = conv(x)
        end.record()
        torch.cuda.synchronize(device)

        latency_ms = start.elapsed_time(end) / num_runs

    del conv, x
    torch.cuda.empty_cache()
    return latency_ms


def main():
    parser = argparse.ArgumentParser(description="cuDNN best-layout baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_runs", type=int, default=500)
    parser.add_argument("--output", default=None,
                        help="CSV output path (default: results/spark_cudnn_layouts.csv)")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "spark_cudnn_layouts.csv")

    print("=" * 78)
    print("cuDNN Best-Layout Baseline (depthwise conv, batch=1, FP16)")
    print("=" * 78)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, CuDNN: {torch.backends.cudnn.version()}")
    print(f"cudnn.benchmark = True (auto-selects fastest algorithm)")
    print(f"Runs per config: {args.num_runs}")
    print()

    header = f"{'Config':<32} {'NCHW (ms)':>10} {'NHWC (ms)':>10} {'Best':>6} {'Ratio':>8}"
    print(header)
    print("-" * 78)

    rows = []
    for name, C, K, H, W in CONFIGS:
        nchw_ms = benchmark_kernel(C, K, H, W, "NCHW", device, args.num_runs)
        nhwc_ms = benchmark_kernel(C, K, H, W, "NHWC", device, args.num_runs)

        best = "NCHW" if nchw_ms < nhwc_ms else "NHWC"
        ratio = max(nchw_ms, nhwc_ms) / min(nchw_ms, nhwc_ms)
        print(f"{name:<32} {nchw_ms:>10.3f} {nhwc_ms:>10.3f} {best:>6} {ratio:>7.2f}x")

        rows.append({
            "config": name,
            "channels": C,
            "kernel_size": K,
            "spatial": f"{H}x{W}",
            "nchw_ms": f"{nchw_ms:.3f}",
            "nhwc_ms": f"{nhwc_ms:.3f}",
            "best_layout": best,
            "ratio": f"{ratio:.2f}",
        })

    print()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
