#!/usr/bin/env python3
"""
RepLKNet-31B in NCHW (contiguous memory format) with stock cuDNN, no custom
kernels. Captures per-DW-layer timing in-model using forward_pre_hook (same
methodology as benchmark_replknet_nhwc_fp16.py).

Tests Hypothesis 2: is in-model cuDNN-NCHW DW time slower than the standalone
per-kernel benchmark (because L2 is disrupted by surrounding PW/BN/GELU ops),
or does it match (cuDNN-NCHW is robust to cache disruption)?

Output format matches the NHWC version so the two can be compared directly:
    Stage Kernel Channels Count Per-call(ms) Total(ms)

Usage:
    python benchmark_replknet_nchw_per_layer.py --device cuda:0
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


def find_dw_layers(model):
    """Find all depthwise conv layers in the RepLKNet model.
    DW layers are nn.Conv2d with groups == in_channels and kernel_size > 1.
    """
    dw_layers = {}  # name -> (stage, kernel_size, channels)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.groups == mod.in_channels and mod.kernel_size[0] > 1:
            ks = mod.kernel_size[0]
            ch = mod.in_channels
            # Extract stage number from name pattern 'stages.{i}...'
            stage = -1
            for tok in name.split('.'):
                if tok.startswith('stages'):
                    continue
                if tok.isdigit():
                    stage = int(tok)
                    break
            if stage == -1 and 'stages' in name:
                parts = name.split('.')
                idx = parts.index('stages')
                if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    stage = int(parts[idx + 1])
            dw_layers[name] = (stage, ks, ch)
    return dw_layers


def time_layer_with_captured_input(mod, inp, n_warmup=5, n_runs=20):
    """Time a module in isolation using a captured input tensor."""
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
        return times[n_runs // 4]  # 25th percentile, matches NHWC script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, cuDNN: {torch.backends.cudnn.version()}")
    print()

    print("Loading RepLKNet-31B (NCHW, stock cuDNN, no custom kernels)...")
    from replknet import create_RepLKNet31B
    CKPT = os.path.join(_THIS_DIR, 'replknet', 'RepLKNet-31B_ImageNet-1K_224.pth')
    model = create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000,
                               use_checkpoint=False, small_kernel_merged=False)
    sd = torch.load(CKPT, map_location='cpu')
    sd = sd.get('state_dict', sd.get('model', sd))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.structural_reparam()
    model.deep_fuse_BN()
    model.eval()
    model = model.to(device).half().to(memory_format=torch.contiguous_format)

    # Identify DW layers
    dw_layers = find_dw_layers(model)
    print(f"Found {len(dw_layers)} depthwise conv layers")
    print()

    # Capture activations
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)  # NCHW

    activations = {}
    handles = []
    for name in dw_layers:
        def make_pre(n):
            def hook(mod, inp):
                activations[n] = inp[0].clone()
            return hook
        mod = dict(model.named_modules())[name]
        handles.append(mod.register_forward_pre_hook(make_pre(name)))

    with torch.no_grad():
        model(x)
        torch.cuda.synchronize(device)

    for h in handles:
        h.remove()

    # Time each DW layer in isolation
    dw_times = {}
    for name, (stage, ks, ch) in dw_layers.items():
        mod = dict(model.named_modules())[name]
        inp = activations[name]
        dw_times[name] = time_layer_with_captured_input(mod, inp)

    # Group by (stage, kernel_size, channels)
    dw_grouped = defaultdict(lambda: {'count': 0, 'total': 0.0, 'per_call': 0.0})
    for name, (stage, ks, ch) in dw_layers.items():
        key = (stage, ks, ch)
        dw_grouped[key]['count'] += 1
        dw_grouped[key]['total'] += dw_times[name]
        dw_grouped[key]['per_call'] = dw_times[name]

    print(f"  Depthwise convolution breakdown (cuDNN-NCHW, in-model):")
    print(f"  {'Stage':<8} {'Kernel':<8} {'Ch':<6} {'Count':<7} {'Per-call':>10} {'Total':>10}")
    print(f"  {'-' * 52}")
    total_dw = 0
    for (stage, ks, ch), info in sorted(dw_grouped.items()):
        pc = info['per_call']
        tot = info['total']
        total_dw += tot
        print(f"  {stage:<8} {ks}x{ks:<5} {ch:<6} {info['count']:<7} {pc:>8.3f} ms {tot:>8.3f} ms")
    print(f"  {'-' * 52}")
    print(f"  {'Total depthwise (NCHW, in-model):':<31} {'':<7} {'':>10} {total_dw:>8.3f} ms")


if __name__ == '__main__':
    main()
