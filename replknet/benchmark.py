#!/usr/bin/env python3
"""
RepLKNet-31B benchmark: FP16 channels_last latency comparison.

Compares CuDNN vs custom depthwise kernels (eager and torch.compile).

Usage:
    python benchmark.py --device cuda:0 --num_runs 100
    python benchmark.py --device cuda:0 --per_layer
    python benchmark.py --device cuda:0 --skip_verify --num_runs 20
"""

import argparse
import os
import sys
import time
from collections import defaultdict
import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPLKNET_REPO = os.path.join(_THIS_DIR, 'RepLKNet-pytorch')
sys.path.insert(0, REPLKNET_REPO)
sys.path.insert(0, os.path.join(_THIS_DIR, 'kernels'))

CHECKPOINT = os.path.join(_THIS_DIR, 'RepLKNet-31B_ImageNet-1K_224.pth')


def load_model(device):
    """Load RepLKNet-31B as FP16 channels_last."""
    from replknet import create_RepLKNet31B

    model = create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000,
                                use_checkpoint=False, small_kernel_merged=False)
    if os.path.exists(CHECKPOINT):
        sd = torch.load(CHECKPOINT, map_location='cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
        elif 'model' in sd:
            sd = sd['model']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    else:
        print(f"WARNING: Checkpoint not found at {CHECKPOINT}")
        print("Running with random weights (latency benchmark only)")

    model.structural_reparam()
    model.deep_fuse_BN()
    model.eval()
    model = model.to(device).half().to(memory_format=torch.channels_last)
    return model


def make_input(device):
    return torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16,
                       ).to(memory_format=torch.channels_last)


def benchmark(model, x, device, num_runs=100, warmup=50):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    torch.cuda.synchronize(device)

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    trim = max(1, len(times) // 20)
    t = times[trim:-trim]
    mean = sum(t) / len(t)
    std = (sum((v - mean) ** 2 for v in t) / len(t)) ** 0.5
    return mean, std, min(t), max(t)


def profile_breakdown(model, x, device, config_name):
    """Isolate each depthwise layer and measure individually."""
    from depthwise_replknet_nhwc_ops import (
        NHWCDepthwise31x31_128, NHWCDepthwise29x29_256, NHWCDepthwise27x27_512
    )
    custom_types = (NHWCDepthwise31x31_128, NHWCDepthwise29x29_256, NHWCDepthwise27x27_512)

    with torch.no_grad():
        for _ in range(10):
            model(x)
    torch.cuda.synchronize(device)

    dw_layers = {}
    for name, mod in model.named_modules():
        if len(list(mod.children())) > 0:
            continue
        stage = -1
        if name.startswith('stages.'):
            parts = name.split('.')
            if len(parts) > 1:
                try:
                    stage = int(parts[1])
                except ValueError:
                    pass

        if isinstance(mod, custom_types):
            if isinstance(mod, NHWCDepthwise31x31_128):
                dw_layers[name] = (stage, 31, 128)
            elif isinstance(mod, NHWCDepthwise29x29_256):
                dw_layers[name] = (stage, 29, 256)
            elif isinstance(mod, NHWCDepthwise27x27_512):
                dw_layers[name] = (stage, 27, 512)
        elif isinstance(mod, nn.Conv2d):
            if mod.groups == mod.in_channels and mod.groups > 1:
                dw_layers[name] = (stage, mod.kernel_size[0], mod.in_channels)

    N = 20
    dw_times = {}
    with torch.no_grad():
        activations = {}
        handles = []
        for name in dw_layers:
            def make_pre(n):
                def hook(mod, inp):
                    activations[n] = inp[0].clone()
                return hook
            mod = dict(model.named_modules())[name]
            handles.append(mod.register_forward_pre_hook(make_pre(name)))

        model(x)
        torch.cuda.synchronize(device)
        for h in handles:
            h.remove()

        for name, (stage, ks, ch) in dw_layers.items():
            mod = dict(model.named_modules())[name]
            inp = activations[name]
            for _ in range(5):
                mod(inp)
            torch.cuda.synchronize(device)

            times = []
            for _ in range(N):
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                mod(inp)
                torch.cuda.synchronize(device)
                times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            dw_times[name] = times[N // 4]

    dw_grouped = defaultdict(lambda: {'count': 0, 'total': 0.0, 'per_call': 0.0})
    for name, (stage, ks, ch) in dw_layers.items():
        key = (stage, ks, ch)
        dw_grouped[key]['count'] += 1
        dw_grouped[key]['total'] += dw_times[name]
        dw_grouped[key]['per_call'] = dw_times[name]

    print(f"\n  Depthwise convolution breakdown:")
    print(f"  {'Stage':<8} {'Kernel':<8} {'Ch':<6} {'Count':<7} {'Per-call':>10} {'Total':>10}")
    print(f"  {'-'*52}")
    total_dw = 0
    for (stage, ks, ch), info in sorted(dw_grouped.items()):
        pc = info['per_call']
        tot = info['total']
        total_dw += tot
        print(f"  {stage:<8} {ks}x{ks:<5} {ch:<6} {info['count']:<7} {pc:>8.3f} ms {tot:>8.3f} ms")
    print(f"  {'-'*52}")
    print(f"  {'Total depthwise:':<31} {'':<7} {'':>10} {total_dw:>8.3f} ms")
    return total_dw


def verify_outputs(model_ref, model_custom, device):
    """Compare reference outputs with custom kernel outputs."""
    x = make_input(device)
    with torch.no_grad():
        out_ref = model_ref(x)
        out_custom = model_custom(x)

    _, pred_ref = out_ref.topk(5)
    _, pred_custom = out_custom.topk(5)
    top1_match = (pred_ref[0, 0] == pred_custom[0, 0]).item()
    top5_match = bool(set(pred_ref[0].tolist()) & set(pred_custom[0].tolist()))
    max_diff = (out_ref.float() - out_custom.float()).abs().max().item()

    print(f"  Top-1 match: {top1_match}")
    print(f"  Top-5 overlap: {top5_match}")
    print(f"  Max logit diff: {max_diff:.4f}")


def main():
    parser = argparse.ArgumentParser(description='RepLKNet-31B benchmark')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--per_layer', action='store_true',
                        help='Profile per-layer depthwise breakdown')
    parser.add_argument('--skip_verify', action='store_true',
                        help='Skip output verification')
    args = parser.parse_args()

    device = args.device
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 70)
    print("RepLKNet-31B Benchmark (FP16 channels_last)")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, CuDNN: {torch.backends.cudnn.version()}")
    print()

    from depthwise_replknet_nhwc_ops import replace_replknet_depthwise_nhwc

    results = {}
    dw_totals = {}

    # Config 1: Custom eager
    print("=" * 70)
    print("Config 1: Custom depthwise kernels (eager)")
    print("=" * 70)
    model = load_model(device)
    model = replace_replknet_depthwise_nhwc(model, verbose=True)
    x = make_input(device)
    mean, std, mn, mx = benchmark(model, x, device, args.num_runs, args.warmup)
    results['custom eager'] = mean
    print(f"  {mean:.2f} +/- {std:.2f} ms  (min {mn:.2f}, max {mx:.2f})")
    if args.per_layer:
        dw_totals['custom eager'] = profile_breakdown(model, x, device, 'custom eager')
    model_custom = model
    del x
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    # Config 2: Custom + compile
    print("\n" + "=" * 70)
    print("Config 2: Custom depthwise + torch.compile")
    print("=" * 70)
    model = load_model(device)
    model = replace_replknet_depthwise_nhwc(model, verbose=False)
    model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
    x = make_input(device)
    mean, std, mn, mx = benchmark(model, x, device, args.num_runs, args.warmup)
    results['custom+compile'] = mean
    print(f"  {mean:.2f} +/- {std:.2f} ms  (min {mn:.2f}, max {mx:.2f})")
    del model, x
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    # Config 3: CuDNN eager
    print("\n" + "=" * 70)
    print("Config 3: CuDNN (eager)")
    print("=" * 70)
    model = load_model(device)
    x = make_input(device)
    mean, std, mn, mx = benchmark(model, x, device, args.num_runs, args.warmup)
    results['CuDNN eager'] = mean
    print(f"  {mean:.2f} +/- {std:.2f} ms  (min {mn:.2f}, max {mx:.2f})")
    if args.per_layer:
        dw_totals['CuDNN eager'] = profile_breakdown(model, x, device, 'CuDNN eager')
    del model, x
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    # Config 4: CuDNN + compile
    print("\n" + "=" * 70)
    print("Config 4: CuDNN + torch.compile")
    print("=" * 70)
    model = load_model(device)
    model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
    x = make_input(device)
    mean, std, mn, mx = benchmark(model, x, device, args.num_runs, args.warmup)
    results['CuDNN+compile'] = mean
    print(f"  {mean:.2f} +/- {std:.2f} ms  (min {mn:.2f}, max {mx:.2f})")
    del model, x
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    # Verification
    if not args.skip_verify:
        print("\n" + "=" * 70)
        print("Verification")
        print("=" * 70)
        model_ref = load_model(device)
        verify_outputs(model_ref, model_custom, device)
        del model_ref

    del model_custom
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (all FP16 NHWC channels_last)")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Latency':>10} {'vs CuDNN':>10}", end='')
    if dw_totals:
        print(f" {'DW total':>10} {'Non-DW':>10}")
    else:
        print()
    print("-" * (70 if dw_totals else 57))
    baseline = results.get('CuDNN eager', 1.0)
    for name in ['CuDNN eager', 'CuDNN+compile', 'custom eager', 'custom+compile']:
        if name in results:
            t = results[name]
            speedup = baseline / t
            line = f"{name:<35} {t:>8.2f} ms {speedup:>9.2f}x"
            if name in dw_totals:
                dw = dw_totals[name]
                non_dw = t - dw
                line += f" {dw:>8.2f} ms {non_dw:>8.2f} ms"
            print(line)
    print("=" * 70)


if __name__ == '__main__':
    main()
