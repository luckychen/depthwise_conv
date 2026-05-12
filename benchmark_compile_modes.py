#!/usr/bin/env python3
"""
End-to-end benchmark: eager / torch.compile (default) / torch.compile (reduce-overhead)
for ConvMixer-1536/20, ConvMixer-768/32, RepLKNet-31B on a single GPU.

Uses CUDA events per-iteration (bulk perf_counter underestimates compiled-model
latency under reduce-overhead by up to 4x — context.md note).

Output: results/{platform}_compile_modes.csv  (platform = 'spark' or 'ada')

Usage:
    python benchmark_compile_modes.py --device cuda:0 --num_runs 500 --output results/spark_compile_modes.csv
"""

import torch
import torch.nn as nn
import argparse
import csv
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'convmixer'))
sys.path.insert(0, os.path.join(_THIS_DIR, 'convmixer', 'kernels'))

from convmixer import load_pretrained
from depthwise_torch_ops import replace_depthwise_convs


def time_cuda_events(fn, x, num_runs=500, warmup=200):
    """Per-iteration CUDA events. Returns mean latency in ms (trimmed 10-90 percentile)."""
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


def make_convmixer(name, custom, device):
    m = load_pretrained(name, checkpoints_dir=os.path.join(_THIS_DIR, 'convmixer', 'checkpoints'),
                        device=device, channels_last=True)
    if custom:
        m = replace_depthwise_convs(m, verbose=False)
    return m.to(dtype=torch.float16, memory_format=torch.channels_last).eval()


def make_replknet(custom, device):
    REPLKNET_DIR = os.path.join(_THIS_DIR, 'replknet')
    sys.path.insert(0, os.path.join(REPLKNET_DIR, 'RepLKNet-pytorch'))
    sys.path.insert(0, os.path.join(REPLKNET_DIR, 'kernels'))
    from replknet import create_RepLKNet31B
    from depthwise_replknet_nhwc_ops import replace_replknet_depthwise_nhwc
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
    m = m.to(device).half().to(memory_format=torch.channels_last)
    if custom:
        m = replace_replknet_depthwise_nhwc(m, verbose=False)
    return m


def bench_model(factory, model_label, device, num_runs, warmup):
    """Bench one (model, kernel-impl) across three compile modes. Returns dict of times."""
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    results = {}

    # --- eager ---
    m = factory()
    print(f"  [{model_label}] eager...", flush=True)
    t = time_cuda_events(lambda inp: m(inp), x, num_runs=num_runs, warmup=warmup)
    results['eager_ms'] = t
    print(f"    {t:.3f} ms")
    del m; torch.cuda.empty_cache()

    # --- torch.compile mode=default ---
    m = factory()
    md = torch.compile(m, mode='default')
    print(f"  [{model_label}] compile (default)...", flush=True)
    t = time_cuda_events(lambda inp: md(inp), x, num_runs=num_runs, warmup=warmup)
    results['compile_ms'] = t
    print(f"    {t:.3f} ms")
    del m, md; torch.cuda.empty_cache()
    torch._dynamo.reset()

    # --- torch.compile mode=reduce-overhead ---
    m = factory()
    try:
        mr = torch.compile(m, mode='reduce-overhead', fullgraph=False)
        print(f"  [{model_label}] compile (reduce-overhead)...", flush=True)
        t = time_cuda_events(lambda inp: mr(inp), x, num_runs=num_runs, warmup=warmup)
        results['reduce_overhead_ms'] = t
        print(f"    {t:.3f} ms")
    except Exception as e:
        print(f"    reduce-overhead FAILED: {type(e).__name__}: {e}")
        results['reduce_overhead_ms'] = float('nan')
    del m
    if 'mr' in dir():
        del mr
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_runs', type=int, default=500)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--output', default='results/compile_modes.csv')
    parser.add_argument('--models', nargs='+',
                        default=['1536_20', '768_32', 'replknet'],
                        help='which models to benchmark')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, cuDNN: {torch.backends.cudnn.version()}")
    print(f"Runs: {args.num_runs} measurement + {args.warmup} warmup")
    print()

    cases = []
    if '1536_20' in args.models:
        cases.append(('ConvMixer-1536/20', 'CuDNN', lambda: make_convmixer('1536_20', custom=False, device=device)))
        cases.append(('ConvMixer-1536/20', 'Custom', lambda: make_convmixer('1536_20', custom=True, device=device)))
    if '768_32' in args.models:
        cases.append(('ConvMixer-768/32', 'CuDNN', lambda: make_convmixer('768_32', custom=False, device=device)))
        cases.append(('ConvMixer-768/32', 'Custom', lambda: make_convmixer('768_32', custom=True, device=device)))
    if 'replknet' in args.models:
        cases.append(('RepLKNet-31B', 'CuDNN', lambda: make_replknet(custom=False, device=device)))
        cases.append(('RepLKNet-31B', 'Custom', lambda: make_replknet(custom=True, device=device)))

    rows = []
    for model_name, impl, factory in cases:
        print(f"=== {model_name} {impl} ===")
        t = bench_model(factory, f"{model_name} {impl}", device, args.num_runs, args.warmup)
        rows.append({
            'model': model_name,
            'impl': impl,
            'eager_ms': t['eager_ms'],
            'compile_ms': t['compile_ms'],
            'reduce_overhead_ms': t['reduce_overhead_ms'],
        })
        print()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'impl', 'eager_ms', 'compile_ms', 'reduce_overhead_ms'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.output}")

    # Print summary table
    print()
    print(f"{'Model':<22} {'Impl':<8} {'Eager':>10} {'Compile':>10} {'r-o':>10} {'r-o gain':>10}")
    print("-" * 78)
    for r in rows:
        gain = r['compile_ms'] / r['reduce_overhead_ms'] if r['reduce_overhead_ms'] == r['reduce_overhead_ms'] else float('nan')
        print(f"{r['model']:<22} {r['impl']:<8} {r['eager_ms']:>9.3f}ms {r['compile_ms']:>9.3f}ms {r['reduce_overhead_ms']:>9.3f}ms {gain:>9.2f}x")


if __name__ == '__main__':
    main()
