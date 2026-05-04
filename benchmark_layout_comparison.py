#!/usr/bin/env python3
"""
E2E layout comparison: NCHW vs NHWC for ConvMixer and RepLKNet with cuDNN
(no custom kernels).

Measures the layout dilemma: for K>=9 cuDNN depthwise prefers NCHW, but pointwise
layers prefer NHWC. For K=7, cuDNN has a specialized NHWC depthwise kernel,
so NHWC wins everywhere.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmark_layout_comparison.py --device cuda:0 --num_runs 500
"""

import torch
import time
import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'convmixer'))
sys.path.insert(0, os.path.join(_THIS_DIR, 'convmixer', 'kernels'))
sys.path.insert(0, os.path.join(_THIS_DIR, 'replknet'))
REPLKNET_REPO = os.path.join(_THIS_DIR, 'replknet', 'RepLKNet-pytorch')
sys.path.insert(0, REPLKNET_REPO)

from convmixer import load_pretrained


def benchmark(model, x, num_runs=500, warmup=200):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_runs):
            model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / num_runs * 1000  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args()

    device = args.device
    print("=" * 70)
    print("E2E Layout Comparison — cuDNN NCHW vs NHWC (no custom kernels)")
    print("=" * 70)
    print(f"Device:   {torch.cuda.get_device_name(device)}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Runs:     {args.num_runs} (warmup={args.warmup})")
    print()

    ckpt_dir = os.path.join(_THIS_DIR, 'convmixer', 'checkpoints')
    results = []

    for model_name, label in [("768_32", "ConvMixer-768/32 (7×7)"),
                               ("1536_20", "ConvMixer-1536/20 (9×9)")]:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        for layout_name, channels_last in [("NHWC", True), ("NCHW", False)]:
            for use_compile in [False, True]:
                compile_label = "+compile" if use_compile else "eager"
                tag = f"cuDNN {layout_name} {compile_label}"
                print(f"\n  [{tag}] Loading...", end="", flush=True)

                m = load_pretrained(model_name, checkpoints_dir=ckpt_dir,
                                    device=device, channels_last=channels_last)
                m = m.to(dtype=torch.float16)
                if channels_last:
                    m = m.to(memory_format=torch.channels_last)
                m.eval()

                if use_compile:
                    m = torch.compile(m, mode='default')

                if channels_last:
                    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(
                        memory_format=torch.channels_last)
                else:
                    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)

                print(" benchmarking...", end="", flush=True)
                latency = benchmark(m, x, num_runs=args.num_runs, warmup=args.warmup)
                print(f"  {latency:.2f} ms")

                results.append({
                    "model": label,
                    "layout": layout_name,
                    "compile": compile_label,
                    "tag": tag,
                    "latency_ms": latency,
                })

                del m
                torch.cuda.empty_cache()

    # RepLKNet-31B
    print(f"\n{'─' * 70}")
    print(f"  RepLKNet-31B (27-31×31)")
    print(f"{'─' * 70}")

    from replknet import create_RepLKNet31B
    replk_ckpt = os.path.join(_THIS_DIR, 'replknet', 'RepLKNet-31B_ImageNet-1K_224.pth')

    for layout_name, channels_last in [("NHWC", True), ("NCHW", False)]:
        for use_compile in [False, True]:
            compile_label = "+compile" if use_compile else "eager"
            tag = f"cuDNN {layout_name} {compile_label}"
            print(f"\n  [{tag}] Loading...", end="", flush=True)

            m = create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000,
                                   use_checkpoint=False, small_kernel_merged=False)
            if os.path.exists(replk_ckpt):
                sd = torch.load(replk_ckpt, map_location='cpu')
                if 'state_dict' in sd:
                    sd = sd['state_dict']
                elif 'model' in sd:
                    sd = sd['model']
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
                m.load_state_dict(sd, strict=False)

            m.structural_reparam()
            m.deep_fuse_BN()
            m.eval()
            m = m.to(device).half()
            if channels_last:
                m = m.to(memory_format=torch.channels_last)

            if use_compile:
                m = torch.compile(m, mode='reduce-overhead')

            if channels_last:
                x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(
                    memory_format=torch.channels_last)
            else:
                x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)

            print(" benchmarking...", end="", flush=True)
            latency = benchmark(m, x, num_runs=min(args.num_runs, 100), warmup=min(args.warmup, 50))
            print(f"  {latency:.2f} ms")

            results.append({
                "model": "RepLKNet-31B (27-31×31)",
                "layout": layout_name,
                "compile": compile_label,
                "tag": tag,
                "latency_ms": latency,
            })

            del m
            torch.cuda.empty_cache()

    print(f"\n\n{'=' * 70}")
    print("SUMMARY — E2E Layout Comparison (cuDNN only, model.half(), B=1)")
    print("=" * 70)
    print(f"{'Model':<28} {'Config':<24} {'Latency':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<28} {r['tag']:<24} {r['latency_ms']:>8.2f} ms")

    # Per-model analysis
    for model_name in ["ConvMixer-768/32 (7×7)", "ConvMixer-1536/20 (9×9)", "RepLKNet-31B (27-31×31)"]:
        model_results = [r for r in results if r["model"] == model_name]
        if len(model_results) == 4:
            nhwc_eager = next(r for r in model_results if r["layout"] == "NHWC" and r["compile"] == "eager")
            nchw_eager = next(r for r in model_results if r["layout"] == "NCHW" and r["compile"] == "eager")
            nhwc_comp = next(r for r in model_results if r["layout"] == "NHWC" and r["compile"] == "+compile")
            nchw_comp = next(r for r in model_results if r["layout"] == "NCHW" and r["compile"] == "+compile")
            print(f"\n  {model_name}:")
            print(f"    Eager:   NHWC {nhwc_eager['latency_ms']:.2f} vs NCHW {nchw_eager['latency_ms']:.2f} ms "
                  f"→ {'NHWC' if nhwc_eager['latency_ms'] < nchw_eager['latency_ms'] else 'NCHW'} wins "
                  f"({max(nhwc_eager['latency_ms'], nchw_eager['latency_ms']) / min(nhwc_eager['latency_ms'], nchw_eager['latency_ms']):.2f}×)")
            print(f"    Compile: NHWC {nhwc_comp['latency_ms']:.2f} vs NCHW {nchw_comp['latency_ms']:.2f} ms "
                  f"→ {'NHWC' if nhwc_comp['latency_ms'] < nchw_comp['latency_ms'] else 'NCHW'} wins "
                  f"({max(nhwc_comp['latency_ms'], nchw_comp['latency_ms']) / min(nhwc_comp['latency_ms'], nchw_comp['latency_ms']):.2f}×)")

    print(f"\n{'=' * 70}")
    print("DONE")


if __name__ == "__main__":
    main()
