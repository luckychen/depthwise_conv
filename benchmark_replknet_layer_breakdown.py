#!/usr/bin/env python3
"""
RepLKNet-31B per-layer-type breakdown: time each module (DW, PW, BN, GELU, etc.)
in-model using forward_pre_hook capture, then aggregate by module type.

Runs in BOTH layouts (NCHW with stock cuDNN; NHWC with custom DW kernels) so
the time-by-layer-type table is directly comparable.

Tests Hypothesis 1: is the in-network NHWC advantage over NCHW for non-DW ops
(PW, BN, GELU) larger than the per-kernel pointwise benchmark would predict?

Usage:
    python benchmark_replknet_layer_breakdown.py --device cuda:0
"""

import argparse
import os
import sys
import time
from collections import defaultdict
import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPLKNET_REPO = os.path.join(_THIS_DIR, 'replknet', 'RepLKNet-pytorch')
sys.path.insert(0, REPLKNET_REPO)
sys.path.insert(0, os.path.join(_THIS_DIR, 'replknet', 'kernels'))


def classify_module(name, mod):
    """Return layer type label, or None if module should be skipped."""
    if isinstance(mod, nn.Conv2d):
        if mod.groups == mod.in_channels and mod.kernel_size[0] > 1:
            return 'DW'
        if mod.kernel_size[0] == 1:
            return 'PW(1x1)'
        return f'Conv{mod.kernel_size[0]}x{mod.kernel_size[0]}'
    if isinstance(mod, nn.BatchNorm2d):
        return 'BN'
    if isinstance(mod, (nn.GELU, nn.ReLU)):
        return 'Activation'
    if isinstance(mod, nn.Linear):
        return 'Linear'
    if isinstance(mod, nn.AdaptiveAvgPool2d):
        return 'AvgPool'
    if isinstance(mod, nn.LayerNorm):
        return 'LayerNorm'
    return None


def time_layer(mod, inp, n_warmup=5, n_runs=20):
    """Time a module in isolation using captured input."""
    with torch.no_grad():
        for _ in range(n_warmup):
            mod(inp)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mod(inp)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        return times[n_runs // 4]


def measure_e2e(model, x, n_warmup=50, n_runs=100):
    """Measure E2E latency with CUDA events per iter."""
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    times.sort()
    return sum(times[10:90]) / 80  # 10-90 percentile trimmed mean


def build_replknet(memory_format, custom_dw, device):
    """Build RepLKNet-31B with the chosen memory format and DW kernels."""
    from replknet import create_RepLKNet31B
    CKPT = os.path.join(_THIS_DIR, 'replknet', 'RepLKNet-31B_ImageNet-1K_224.pth')
    m = create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000,
                           use_checkpoint=False, small_kernel_merged=False)
    sd = torch.load(CKPT, map_location='cpu')
    sd = sd.get('state_dict', sd.get('model', sd))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.structural_reparam()
    m.deep_fuse_BN()
    m.eval()
    m = m.to(device).half().to(memory_format=memory_format)
    if custom_dw:
        from depthwise_replknet_nhwc_ops import replace_replknet_depthwise_nhwc
        m = replace_replknet_depthwise_nhwc(m, verbose=False)
    return m


def profile_one_config(model, x, device, label):
    """Time every module-of-interest in-model. Return dict by type."""
    # E2E first
    e2e = measure_e2e(model, x)
    print(f"  [{label}] E2E: {e2e:.3f} ms")

    # Identify modules to time
    targets = []  # list of (name, mod, type_label)
    for name, mod in model.named_modules():
        t = classify_module(name, mod)
        if t is not None:
            targets.append((name, mod, t))

    # Hook each target to capture input
    activations = {}
    handles = []
    for name, mod, _ in targets:
        def make_pre(n):
            def hook(m, inp):
                if isinstance(inp, (tuple, list)) and len(inp) > 0:
                    activations[n] = inp[0].clone() if isinstance(inp[0], torch.Tensor) else None
            return hook
        handles.append(mod.register_forward_pre_hook(make_pre(name)))

    with torch.no_grad():
        model(x)
        torch.cuda.synchronize(device)
    for h in handles:
        h.remove()

    # Time each module that captured a valid input
    by_type = defaultdict(lambda: {'count': 0, 'total_ms': 0.0})
    skipped = 0
    for name, mod, ttype in targets:
        if name not in activations or activations[name] is None:
            skipped += 1
            continue
        try:
            ms = time_layer(mod, activations[name])
            by_type[ttype]['count'] += 1
            by_type[ttype]['total_ms'] += ms
        except Exception as e:
            skipped += 1

    print(f"  [{label}] Profiled {sum(v['count'] for v in by_type.values())} modules, skipped {skipped}")
    return {'e2e_ms': e2e, 'by_type': dict(by_type)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='results/replknet_layer_breakdown.txt')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, cuDNN: {torch.backends.cudnn.version()}")
    print()

    results = {}

    # --- NCHW path (stock cuDNN) ---
    print("=== RepLKNet-31B NCHW (stock cuDNN, contiguous) ===")
    m_nchw = build_replknet(torch.contiguous_format, custom_dw=False, device=device)
    x_nchw = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    results['NCHW'] = profile_one_config(m_nchw, x_nchw, device, 'NCHW')
    del m_nchw
    torch.cuda.empty_cache()
    print()

    # --- NHWC path (custom DW) ---
    print("=== RepLKNet-31B NHWC (custom DW kernels, channels_last) ===")
    m_nhwc = build_replknet(torch.channels_last, custom_dw=True, device=device)
    x_nhwc = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    results['NHWC (Custom DW)'] = profile_one_config(m_nhwc, x_nhwc, device, 'NHWC')
    del m_nhwc
    torch.cuda.empty_cache()
    print()

    # --- Print comparison ---
    all_types = set()
    for r in results.values():
        all_types.update(r['by_type'].keys())
    all_types = sorted(all_types)

    print()
    print(f"{'Layer type':<14} | " + " | ".join(f"{k:>22}" for k in results.keys()))
    print(f"{'-'*14} | " + " | ".join(f"{'-'*22}" for _ in results.keys()))
    for ttype in all_types:
        row = f"{ttype:<14} | "
        cells = []
        for path, r in results.items():
            info = r['by_type'].get(ttype, {'count': 0, 'total_ms': 0.0})
            cells.append(f"{info['count']:3d}x, {info['total_ms']:>7.3f}ms")
        row += " | ".join(f"{c:>22}" for c in cells)
        print(row)
    print(f"{'-'*14} | " + " | ".join(f"{'-'*22}" for _ in results.keys()))
    sum_row = f"{'Sum profiled':<14} | "
    for path, r in results.items():
        s = sum(v['total_ms'] for v in r['by_type'].values())
        sum_row += f"{'':<8}{s:>7.3f}ms      | "
    print(sum_row.rstrip(' |'))

    e2e_row = f"{'E2E (model)':<14} | "
    for path, r in results.items():
        e2e_row += f"{'':<8}{r['e2e_ms']:>7.3f}ms      | "
    print(e2e_row.rstrip(' |'))

    impl_row = f"{'Implicit/other':<14} | "
    for path, r in results.items():
        s = sum(v['total_ms'] for v in r['by_type'].values())
        impl_row += f"{'':<8}{r['e2e_ms']-s:>7.3f}ms      | "
    print(impl_row.rstrip(' |'))

    # Save to file
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(f"Device: {torch.cuda.get_device_name(device)}\n")
        f.write(f"PyTorch: {torch.__version__}, cuDNN: {torch.backends.cudnn.version()}\n\n")
        for path, r in results.items():
            f.write(f"=== {path} ===\n")
            f.write(f"E2E: {r['e2e_ms']:.3f} ms\n")
            for ttype in sorted(r['by_type'].keys()):
                info = r['by_type'][ttype]
                f.write(f"  {ttype:<14} count={info['count']:3d}  total={info['total_ms']:.3f} ms  per-call={info['total_ms']/max(info['count'],1):.4f} ms\n")
            s = sum(v['total_ms'] for v in r['by_type'].values())
            f.write(f"  Sum profiled: {s:.3f} ms;  Implicit/other (residual, dropout, etc): {r['e2e_ms']-s:.3f} ms\n\n")

    print(f"\nWrote {args.output}")


if __name__ == '__main__':
    main()
