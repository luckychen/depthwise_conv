#!/usr/bin/env python3
"""
Batch sweep benchmark: cuDNN-NCHW, cuDNN-NHWC, Triton E2E, Custom.
ConvMixer-1536/20, model.half(), CUDA events per-iteration timing.

Usage:
    export CUDA_VISIBLE_DEVICES=1 && conda activate cuda
    python benchmark_batch_sweep.py --device cuda:0 --num_runs 200

Outputs: results/ada_batch_sweep_all_methods.csv + .log
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import argparse
import csv
import os
import sys
import time

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_DIR, 'convmixer'))
sys.path.insert(0, os.path.join(_REPO_DIR, 'convmixer', 'kernels'))

from convmixer import load_pretrained
from depthwise_torch_ops import replace_depthwise_convs


# ---------------------------------------------------------------------------
# Triton kernel (copy from benchmark_triton.py — per-program NHWC depthwise)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 32},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=8, num_stages=2),
    ],
    key=["H", "W", "C", "K"],
)
@triton.jit
def depthwise_conv2d_nhwc_kernel(
    input_ptr, weight_ptr, output_ptr,
    H, W, C, K, PAD,
    stride_h, stride_w,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_c_blocks = tl.cdiv(C, BLOCK_C)
    hw_idx  = pid // num_c_blocks
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
                    w_off  = kh * K * C + kw * C + c_offsets
                    w_vals = tl.load(weight_ptr + w_off, mask=c_mask, other=0.0).to(tl.float32)
                    acc += x_vals * w_vals

    out_off = oh * stride_h + ow * stride_w + c_offsets
    tl.store(output_ptr + out_off, acc.to(tl.float16), mask=c_mask)


def _triton_depthwise_single(x_hwc, weight_hwc, K, pad):
    """x_hwc: [H, W, C] fp16.  weight_hwc: [K, K, C] fp16.  Returns [H, W, C]."""
    H, W, C = x_hwc.shape
    output = torch.empty_like(x_hwc)
    grid = lambda meta: (H * W * triton.cdiv(C, meta["BLOCK_C"]),)
    depthwise_conv2d_nhwc_kernel[grid](
        x_hwc, weight_hwc, output,
        H, W, C, K, pad,
        W * C, C,
    )
    return output


class TritonDepthwiseConv2d(nn.Module):
    """Wraps Triton kernel as a drop-in for nn.Conv2d (depthwise).
    Accepts channels_last [B, C, H, W] FP16 input."""

    def __init__(self, src_module: nn.Conv2d):
        super().__init__()
        C  = src_module.in_channels
        K  = src_module.kernel_size[0]
        pad = src_module.padding[0] if isinstance(src_module.padding, tuple) \
              else (src_module.padding if isinstance(src_module.padding, int) else K // 2)
        # Weight: [C,1,K,K] → [K,K,C]
        w_hwc = src_module.weight.squeeze(1).permute(1, 2, 0).contiguous().half()
        self.weight = nn.Parameter(w_hwc)
        if src_module.bias is not None:
            self.bias = nn.Parameter(src_module.bias.half())
        else:
            self.register_parameter('bias', None)
        self.K   = K
        self.pad = pad

    def forward(self, x):
        # x: [B, C, H, W] channels_last  →  physical [B, H, W, C]
        B, C, H, W = x.shape
        # Permute to explicit [B, H, W, C] contiguous slice for Triton
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()
        out_bhwc = torch.empty_like(x_bhwc)
        for b in range(B):
            out_bhwc[b] = _triton_depthwise_single(x_bhwc[b], self.weight, self.K, self.pad)
        out = out_bhwc.permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


def replace_with_triton(model):
    """Replace all depthwise Conv2d layers with TritonDepthwiseConv2d."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d) and \
                module.groups == module.in_channels == module.out_channels:
            triton_conv = TritonDepthwiseConv2d(module).to(module.weight.device)
            parent_name = '.'.join(name.split('.')[:-1])
            child_name  = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, triton_conv)
    return model


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def make_input(batch, device, channels_last=True, fp16=True):
    x = torch.randn(batch, 3, 224, 224, device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    if fp16:
        x = x.half()
    return x


def benchmark_cuda_events(model, x, num_runs=200, warmup=50):
    """Per-iteration CUDA events timing. Returns (latency_ms, throughput_img_s)."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            model(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))

    times_sorted = sorted(times)
    # Drop top/bottom 5% to reduce noise
    trim = max(1, num_runs // 20)
    trimmed = times_sorted[trim:-trim]
    latency_ms = sum(trimmed) / len(trimmed)
    throughput  = x.shape[0] / (latency_ms / 1000.0)
    return latency_ms, throughput


def load_model(method, model_name, device, compile_mode='default'):
    """Load ConvMixer with the requested method."""
    if method == 'cudnn_nchw':
        # Standard NCHW (no channels_last)
        import importlib
        from convmixer import load_pretrained as _lp
        m = _lp(model_name,
                checkpoints_dir=os.path.join(_REPO_DIR, 'convmixer', 'checkpoints'),
                device=device, channels_last=False)
        m = m.half().eval()

    elif method == 'cudnn_nhwc':
        m = load_pretrained(model_name,
                            checkpoints_dir=os.path.join(_REPO_DIR, 'convmixer', 'checkpoints'),
                            device=device, channels_last=True)
        m = m.to(dtype=torch.float16, memory_format=torch.channels_last).eval()

    elif method == 'triton':
        m = load_pretrained(model_name,
                            checkpoints_dir=os.path.join(_REPO_DIR, 'convmixer', 'checkpoints'),
                            device=device, channels_last=True)
        m = replace_with_triton(m)
        m = m.to(dtype=torch.float16, memory_format=torch.channels_last).eval()

    elif method == 'custom':
        m = load_pretrained(model_name,
                            checkpoints_dir=os.path.join(_REPO_DIR, 'convmixer', 'checkpoints'),
                            device=device, channels_last=True)
        m = replace_depthwise_convs(m, verbose=False)
        m = m.to(dtype=torch.float16, memory_format=torch.channels_last).eval()

    else:
        raise ValueError(f"Unknown method: {method}")

    if compile_mode is not None:
        print(f"    torch.compile(mode='{compile_mode}')...", flush=True)
        m = torch.compile(m, mode=compile_mode)

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = [
    ('cudnn_nhwc',  'cuDNN NHWC + compile',  True),
    ('cudnn_nchw',  'cuDNN NCHW + compile',  True),
    ('triton',      'Triton + compile',       True),
    ('custom',      'Custom + compile',       True),
]

BATCH_PER_METHOD = {
    'cudnn_nchw': [1, 4, 8],
    'cudnn_nhwc': [1, 4, 8],
    'triton':     [1, 4],
    'custom':     [1, 4, 8],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    default='1536_20',  choices=['1536_20', '768_32'])
    parser.add_argument('--device',   default='cuda:0')
    parser.add_argument('--num_runs', type=int, default=200)
    parser.add_argument('--output',   default=None)
    parser.add_argument('--methods',  nargs='+',
                        default=['cudnn_nhwc', 'cudnn_nchw', 'triton', 'custom'],
                        help='Methods to benchmark')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    out_base = args.output or os.path.join(_REPO_DIR, 'results', 'ada_batch_sweep_all_methods')
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    csv_path = out_base + '.csv'
    log_path = out_base + '.log'

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    # Clear log
    open(log_path, 'w').close()

    log("=" * 72)
    log(f"Batch Sweep: All Methods — ConvMixer-{args.model}")
    log("=" * 72)
    log(f"Device:  {torch.cuda.get_device_name(device)}")
    log(f"PyTorch: {torch.__version__}  Triton: {triton.__version__}")
    log(f"Runs: {args.num_runs} (trimmed ±5%)")
    log("")

    rows = []

    for key, label, use_compile in METHODS:
        if key not in args.methods:
            continue

        batches = BATCH_PER_METHOD[key]
        compile_mode = 'default' if use_compile else None

        log(f"\n[{label}]")
        log(f"  Loading model...")

        try:
            model = load_model(key, args.model, device, compile_mode=compile_mode)
        except Exception as exc:
            log(f"  ERROR loading model: {exc}")
            for b in batches:
                rows.append({'method': key, 'label': label, 'batch': b,
                             'latency_ms': 'ERROR', 'throughput_img_s': 'ERROR'})
            continue

        # Warm up compile at B=1
        log("  Warming up compile (B=1)...")
        try:
            x_warm = make_input(1, device, channels_last=(key != 'cudnn_nchw'))
            with torch.no_grad():
                for _ in range(5):
                    model(x_warm)
            torch.cuda.synchronize()
        except Exception as exc:
            log(f"  ERROR during compile warmup: {exc}")

        for b in batches:
            channels_last = (key != 'cudnn_nchw')
            x = make_input(b, device, channels_last=channels_last)

            log(f"  Batch={b} ...", )
            try:
                lat, thr = benchmark_cuda_events(model, x, num_runs=args.num_runs)
                log(f"    latency={lat:.3f} ms   throughput={thr:.1f} img/s")
                rows.append({'method': key, 'label': label, 'batch': b,
                             'latency_ms': f'{lat:.3f}',
                             'throughput_img_s': f'{thr:.1f}'})
            except Exception as exc:
                log(f"    ERROR: {exc}")
                rows.append({'method': key, 'label': label, 'batch': b,
                             'latency_ms': 'ERROR', 'throughput_img_s': 'ERROR'})

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Write CSV
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['method', 'label', 'batch',
                                                    'latency_ms', 'throughput_img_s'])
            writer.writeheader()
            writer.writerows(rows)
        log(f"\nResults saved to {csv_path}")

    # Print summary table
    log("\n" + "=" * 72)
    log("SUMMARY")
    log("=" * 72)
    log(f"{'Method':<28} {'B':>4} {'Latency':>12} {'Throughput':>14}")
    log("-" * 60)
    for r in rows:
        log(f"{r['label']:<28} {r['batch']:>4} {r['latency_ms']:>11} ms "
            f"{r['throughput_img_s']:>12} img/s")

    log("\nDone.")


if __name__ == '__main__':
    main()
