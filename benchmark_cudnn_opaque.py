#!/usr/bin/env python3
"""
Experiment: What happens when cuDNN depthwise is wrapped as a torch.library
opaque op, preventing Inductor from decomposing it?

This tests whether M2 (deployment-pipeline survival) is purely about avoiding
Inductor decomposition, or whether there are deeper effects.

Configurations tested per model:
  1. cuDNN-NHWC default  (standard nn.Conv2d, channels_last) — Inductor decomposes
  2. cuDNN-NHWC opaque   (torch.library-wrapped cuDNN) — survives torch.compile
  3. cuDNN-NCHW default  (standard nn.Conv2d, contiguous) — Inductor decomposes
  4. cuDNN-NCHW opaque   (torch.library-wrapped cuDNN) — survives torch.compile

Each configuration: eager / compile(default) / compile(reduce-overhead)

Usage:
    python benchmark_cudnn_opaque.py --device cuda:0 --num_runs 500
    python benchmark_cudnn_opaque.py --device cuda:0 --num_runs 500 --output results/cudnn_opaque.csv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import csv
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'convmixer'))

# ─── Register cuDNN depthwise as opaque torch.library op ───

_lib = torch.library.Library("cudnn_opaque", "DEF")
_lib.define("depthwise_conv2d(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int groups) -> Tensor")

@torch.library.impl(_lib, "depthwise_conv2d", "CUDA")
def _cudnn_dw_impl(input, weight, bias, stride, padding, groups):
    return F.conv2d(input, weight, bias, stride=stride, padding=padding, groups=groups)

@torch.library.impl(_lib, "depthwise_conv2d", "Meta")
def _cudnn_dw_meta(input, weight, bias, stride, padding, groups):
    return F.conv2d(input, weight, bias, stride=stride, padding=padding, groups=groups)


class OpaqueCuDNNDepthwise(nn.Module):
    """Drop-in replacement for nn.Conv2d (depthwise) that calls cuDNN
    through a torch.library opaque op, invisible to Inductor."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        assert conv.groups == conv.in_channels, "Expected depthwise conv"
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride = conv.stride
        self.padding = conv.padding
        self.groups = conv.groups

    def forward(self, x):
        return torch.ops.cudnn_opaque.depthwise_conv2d(
            x, self.weight, self.bias,
            list(self.stride), list(self.padding), self.groups
        )


def replace_dw_with_opaque(model):
    """Replace all depthwise Conv2d in-place with OpaqueCuDNNDepthwise."""
    count = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Conv2d) and mod.groups == mod.in_channels and mod.kernel_size[0] > 1:
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], OpaqueCuDNNDepthwise(mod))
            count += 1
    return model, count


# ─── Timing ───

def time_cuda_events(fn, x, num_runs=500, warmup=200):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    times.sort()
    lo = int(num_runs * 0.10)
    hi = int(num_runs * 0.90)
    return sum(times[lo:hi]) / (hi - lo)


# ─── Model builders ───

def build_convmixer(name, memory_format, opaque_dw, device):
    from convmixer import load_pretrained
    m = load_pretrained(name, checkpoints_dir=os.path.join(_THIS_DIR, 'convmixer', 'checkpoints'),
                        device='cpu', channels_last=False)
    m = m.to(dtype=torch.float16, memory_format=memory_format).to(device)
    if opaque_dw:
        m, n = replace_dw_with_opaque(m)
        print(f"    Replaced {n} DW layers with opaque cuDNN")
    m.eval()
    return m


def build_replknet(memory_format, opaque_dw, device):
    REPLKNET_DIR = os.path.join(_THIS_DIR, 'replknet')
    sys.path.insert(0, os.path.join(REPLKNET_DIR, 'RepLKNet-pytorch'))
    from replknet import create_RepLKNet31B
    CKPT = os.path.join(REPLKNET_DIR, 'RepLKNet-31B_ImageNet-1K_224.pth')
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
    if opaque_dw:
        m, n = replace_dw_with_opaque(m)
        print(f"    Replaced {n} DW layers with opaque cuDNN")
    return m


# ─── Benchmark one config ───

def bench_config(factory, label, device, num_runs, warmup):
    fmt = factory._memory_format if hasattr(factory, '_memory_format') else torch.channels_last
    if fmt == torch.channels_last:
        x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    else:
        x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)

    results = {}

    # eager
    m = factory()
    print(f"  [{label}] eager...", flush=True)
    t = time_cuda_events(lambda inp: m(inp), x, num_runs=num_runs, warmup=warmup)
    results['eager_ms'] = t
    print(f"    {t:.3f} ms")
    del m; torch.cuda.empty_cache()

    # torch.compile default
    m = factory()
    mc = torch.compile(m, mode='default')
    print(f"  [{label}] compile (default)...", flush=True)
    t = time_cuda_events(lambda inp: mc(inp), x, num_runs=num_runs, warmup=warmup)
    results['compile_ms'] = t
    print(f"    {t:.3f} ms")
    del m, mc; torch.cuda.empty_cache()
    torch._dynamo.reset()

    # torch.compile reduce-overhead
    m = factory()
    try:
        mr = torch.compile(m, mode='reduce-overhead', fullgraph=False)
        print(f"  [{label}] compile (reduce-overhead)...", flush=True)
        t = time_cuda_events(lambda inp: mr(inp), x, num_runs=num_runs, warmup=warmup)
        results['ro_ms'] = t
        print(f"    {t:.3f} ms")
    except Exception as e:
        print(f"    reduce-overhead FAILED: {e}")
        results['ro_ms'] = float('nan')
    del m
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_runs', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--output', default='')
    parser.add_argument('--models', nargs='+',
                        default=['768_32', '1536_20', 'replknet'])
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, cuDNN: {torch.backends.cudnn.version()}")
    print(f"Runs: {args.num_runs} + {args.warmup} warmup\n")

    rows = []

    for model_key in args.models:
        # Define configs: (label, memory_format, opaque_dw)
        configs = [
            ('NHWC default',  torch.channels_last,   False),
            ('NHWC opaque',   torch.channels_last,   True),
            ('NCHW default',  torch.contiguous_format, False),
            ('NCHW opaque',   torch.contiguous_format, True),
        ]

        for cfg_label, mem_fmt, opaque in configs:
            label = f"{model_key} cuDNN-{cfg_label}"
            print(f"\n=== {label} ===")

            def make_factory(mk=model_key, mf=mem_fmt, op=opaque):
                def factory():
                    if mk in ('768_32', '1536_20'):
                        return build_convmixer(mk, mf, op, device)
                    else:
                        return build_replknet(mf, op, device)
                factory._memory_format = mf
                return factory

            try:
                r = bench_config(make_factory(), label, device, args.num_runs, args.warmup)
                rows.append({
                    'model': model_key,
                    'layout': 'NHWC' if mem_fmt == torch.channels_last else 'NCHW',
                    'opaque': opaque,
                    **r
                })
            except Exception as e:
                print(f"  FAILED: {e}")
                rows.append({
                    'model': model_key,
                    'layout': 'NHWC' if mem_fmt == torch.channels_last else 'NCHW',
                    'opaque': opaque,
                    'eager_ms': float('nan'),
                    'compile_ms': float('nan'),
                    'ro_ms': float('nan'),
                })

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<18} {'Config':<20} {'Eager':>10} {'Compile':>10} {'R-O':>10}")
    print("-" * 90)
    for r in rows:
        tag = f"{'OPAQUE' if r['opaque'] else 'default'}"
        cfg = f"{r['layout']} {tag}"
        print(f"{r['model']:<18} {cfg:<20} {r['eager_ms']:>10.3f} {r['compile_ms']:>10.3f} {r.get('ro_ms', float('nan')):>10.3f}")
    print("=" * 90)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['model', 'layout', 'opaque', 'eager_ms', 'compile_ms', 'ro_ms'])
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.output}")


if __name__ == '__main__':
    main()
