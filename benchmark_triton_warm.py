#!/usr/bin/env python3
"""Matched warm Triton baseline for the manuscript's selected configurations.

Uses the same FP16 NHWC shapes and 500-warmup/500-timed CUDA-event protocol as
the current isolated-kernel measurements. The underlying Triton kernel and its
six-way autotuning space are defined in benchmark_triton.py.
"""

import argparse
import csv
import os

import torch
import triton

from benchmark_triton import benchmark_triton_kernel, verify_correctness


SELECTED_CONFIGS = [
    ("ConvMixer 7x7/768ch/32x32", 768, 7, 32, 32),
    ("ConvMixer 9x9/1536ch/32x32", 1536, 9, 32, 32),
    ("RepLKNet 31x31/128ch/56x56", 128, 31, 56, 56),
    ("RepLKNet 29x29/256ch/28x28", 256, 29, 28, 28),
    ("RepLKNet 27x27/512ch/14x14", 512, 27, 14, 14),
]


def main():
    parser = argparse.ArgumentParser(description="Warm Triton depthwise baseline")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--num-runs", type=int, default=500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 78)
    print("Warm Triton Depthwise Conv2d Baseline (batch=1, FP16, NHWC)")
    print("=" * 78)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}")
    print(f"Warmup/timed: {args.warmup}/{args.num_runs}")
    print()

    rows = []
    for name, channels, kernel, height, width in SELECTED_CONFIGS:
        max_error = verify_correctness(channels, kernel, height, width, device)
        if max_error >= 1.0:
            raise RuntimeError(f"Correctness check failed for {name}: {max_error:.4f}")

        latency_ms = benchmark_triton_kernel(
            channels,
            kernel,
            height,
            width,
            device,
            num_runs=args.num_runs,
            warmup=args.warmup,
        )
        latency_us = latency_ms * 1000
        print(f"{name:<32} {latency_us:>9.1f} us  max_error={max_error:.4f}")
        rows.append(
            {
                "config": name,
                "channels": channels,
                "kernel_size": kernel,
                "spatial": f"{height}x{width}",
                "warmup": args.warmup,
                "timed": args.num_runs,
                "max_abs_error": f"{max_error:.4f}",
                "triton_warm_us": f"{latency_us:.3f}",
            }
        )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
