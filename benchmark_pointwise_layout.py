#!/usr/bin/env python3
"""Benchmark pointwise (1x1) convolution: NCHW vs NHWC layout.

Measures the same configs as Table 1 (tab:pointwise_layout) in the paper.
Run on each platform to get cross-platform pointwise layout comparison.

Usage:
    python benchmark_pointwise_layout.py --device cuda:0 --num_runs 500
"""

import torch
import torch.nn as nn
import argparse
import csv
import os

# (name, in_channels, out_channels, spatial_h, spatial_w)
CONFIGS = [
    ("ConvMixer-1536 (32²×1536)",         1536, 1536, 32, 32),
    ("RepLKNet stg2→3 (14², 512→1024)",    512, 1024, 14, 14),
    ("RepLKNet stg3 (7²×1024)",           1024, 1024,  7,  7),
    ("ConvMixer-768 (32²×768)",             768,  768, 32, 32),
]


def benchmark_pw(in_ch, out_ch, H, W, layout, device, num_runs=500, warmup=200):
    conv = nn.Conv2d(in_ch, out_ch, 1, bias=False).to(device).half()
    if layout == "NHWC":
        conv = conv.to(memory_format=torch.channels_last)
        x = torch.randn(1, in_ch, H, W, device=device, dtype=torch.float16).to(
            memory_format=torch.channels_last)
    else:
        x = torch.randn(1, in_ch, H, W, device=device, dtype=torch.float16).contiguous()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_runs", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        f"pointwise_layout_{torch.cuda.get_device_name(device).replace(' ', '_').lower()}.csv")

    print("=" * 72)
    print(f"Pointwise (1×1) Conv: NCHW vs NHWC (batch=1, FP16)")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, cuDNN: {torch.backends.cudnn.version()}")
    print("=" * 72)
    print(f"{'Config':<38} {'NCHW (ms)':>10} {'NHWC (ms)':>10} {'Speedup':>8}")
    print("-" * 72)

    results = []
    for name, in_ch, out_ch, H, W in CONFIGS:
        nchw_ms = benchmark_pw(in_ch, out_ch, H, W, "NCHW", device, args.num_runs)
        nhwc_ms = benchmark_pw(in_ch, out_ch, H, W, "NHWC", device, args.num_runs)
        speedup = nchw_ms / nhwc_ms
        print(f"{name:<38} {nchw_ms:>10.4f} {nhwc_ms:>10.4f} {speedup:>7.2f}×")
        results.append((name, in_ch, out_ch, H, W, nchw_ms, nhwc_ms, speedup))

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "in_ch", "out_ch", "spatial", "nchw_ms", "nhwc_ms", "speedup"])
        for name, in_ch, out_ch, H, W, nchw, nhwc, spd in results:
            w.writerow([name, in_ch, out_ch, f"{H}x{W}", f"{nchw:.4f}", f"{nhwc:.4f}", f"{spd:.2f}"])

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
