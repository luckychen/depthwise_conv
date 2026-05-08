#!/usr/bin/env python3
"""
RepLKNet-31B batch sweep: cuDNN-NHWC, cuDNN-NCHW, Triton, Custom.
model.half(), CUDA events per-iteration timing, mode=default compile.

Usage:
    export CUDA_VISIBLE_DEVICES=1 && conda activate cuda
    python benchmark_replknet_batch_sweep.py --device cuda:0 --num_runs 100

Output: results/ada_replknet_batch_sweep.csv + .log
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import argparse
import csv
import os
import sys

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
REPLKNET_DIR = os.path.join(_REPO_DIR, 'replknet')
sys.path.insert(0, os.path.join(REPLKNET_DIR, 'RepLKNet-pytorch'))
sys.path.insert(0, os.path.join(REPLKNET_DIR, 'kernels'))

CHECKPOINT = os.path.join(REPLKNET_DIR, 'RepLKNet-31B_ImageNet-1K_224.pth')


# ---------------------------------------------------------------------------
# Triton kernel
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
    H, W, C, K, PAD, stride_h, stride_w,
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


def _triton_dw_single(x_hwc, weight_hwc, K, pad):
    H, W, C = x_hwc.shape
    output = torch.empty_like(x_hwc)
    grid = lambda meta: (H * W * triton.cdiv(C, meta["BLOCK_C"]),)
    depthwise_conv2d_nhwc_kernel[grid](
        x_hwc, weight_hwc, output, H, W, C, K, pad, W * C, C)
    return output


class TritonDepthwiseConv2d(nn.Module):
    def __init__(self, src: nn.Conv2d):
        super().__init__()
        K   = src.kernel_size[0]
        pad = src.padding[0] if isinstance(src.padding, tuple) else \
              (src.padding if isinstance(src.padding, int) else K // 2)
        w_hwc = src.weight.squeeze(1).permute(1, 2, 0).contiguous().half()
        self.weight = nn.Parameter(w_hwc)
        self.bias = nn.Parameter(src.bias.half()) if src.bias is not None else None
        self.K = K
        self.pad = pad

    def forward(self, x):
        B, C, H, W = x.shape
        x_bhwc  = x.permute(0, 2, 3, 1).contiguous()
        out_bhwc = torch.empty_like(x_bhwc)
        for b in range(B):
            out_bhwc[b] = _triton_dw_single(x_bhwc[b], self.weight, self.K, self.pad)
        out = out_bhwc.permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


def replace_with_triton(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d) and \
                module.groups == module.in_channels == module.out_channels:
            t_conv = TritonDepthwiseConv2d(module).to(module.weight.device)
            parent_name = '.'.join(name.split('.')[:-1])
            child_name  = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, t_conv)
    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_base_model(device, channels_last=True):
    from replknet import create_RepLKNet31B
    model = create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000,
                                use_checkpoint=False, small_kernel_merged=False)
    if os.path.exists(CHECKPOINT):
        sd = torch.load(CHECKPOINT, map_location='cpu')
        sd = sd.get('state_dict', sd.get('model', sd))
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    model.structural_reparam()
    model.deep_fuse_BN()
    model.eval()
    if channels_last:
        model = model.to(device).half().to(memory_format=torch.channels_last)
    else:
        model = model.to(device).half()
    return model


def load_model(method, device, compile_mode='default'):
    if method == 'cudnn_nhwc':
        m = _load_base_model(device, channels_last=True)

    elif method == 'cudnn_nchw':
        m = _load_base_model(device, channels_last=False)

    elif method == 'triton':
        m = _load_base_model(device, channels_last=True)
        m = replace_with_triton(m)

    elif method == 'custom':
        from depthwise_replknet_nhwc_ops import replace_replknet_depthwise_nhwc
        m = _load_base_model(device, channels_last=True)
        m = replace_replknet_depthwise_nhwc(m, verbose=False)

    m.eval()

    if compile_mode is not None:
        print(f"    torch.compile(mode='{compile_mode}')...", flush=True)
        m = torch.compile(m, mode=compile_mode, fullgraph=False)

    return m


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def make_input(batch, device, channels_last=True):
    x = torch.randn(batch, 3, 224, 224, device=device, dtype=torch.float16)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    return x


def benchmark_cuda_events(model, x, num_runs=100, warmup=30):
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

    times_s = sorted(times)
    trim = max(1, num_runs // 20)
    trimmed = times_s[trim:-trim]
    lat_ms = sum(trimmed) / len(trimmed)
    thr    = x.shape[0] / (lat_ms / 1000.0)
    return lat_ms, thr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METHODS = ['cudnn_nhwc', 'cudnn_nchw', 'triton', 'custom']

BATCH_PER_METHOD = {
    'cudnn_nhwc': [1, 2, 4],
    'cudnn_nchw': [1, 2, 4],
    'triton':     [1, 2],
    'custom':     [1, 2, 4],
}

LABELS = {
    'cudnn_nhwc': 'cuDNN NHWC + compile',
    'cudnn_nchw': 'cuDNN NCHW + compile',
    'triton':     'Triton + compile',
    'custom':     'Custom + compile',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',   default='cuda:0')
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--output',   default=None)
    parser.add_argument('--methods',  nargs='+', default=METHODS)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    out_base = args.output or os.path.join(_REPO_DIR, 'results', 'ada_replknet_batch_sweep')
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    csv_path = out_base + '.csv'
    log_path = out_base + '.log'
    open(log_path, 'w').close()

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log("=" * 72)
    log("RepLKNet-31B Batch Sweep — Ada (RTX 6000)")
    log("=" * 72)
    log(f"Device:  {torch.cuda.get_device_name(device)}")
    log(f"PyTorch: {torch.__version__}  Triton: {triton.__version__}")
    log(f"Runs: {args.num_runs} (trimmed ±5%)")
    log("")

    rows = []

    for key in args.methods:
        if key not in METHODS:
            continue
        label   = LABELS[key]
        batches = BATCH_PER_METHOD[key]

        log(f"\n[{label}]")
        log("  Loading model...")

        try:
            model = load_model(key, device, compile_mode='default')
        except Exception as exc:
            log(f"  ERROR loading: {exc}")
            for b in batches:
                rows.append({'method': key, 'label': label, 'batch': b,
                             'latency_ms': 'ERROR', 'throughput_img_s': 'ERROR'})
            continue

        # Warm up compile
        try:
            cl = (key != 'cudnn_nchw')
            x_w = make_input(1, device, channels_last=cl)
            with torch.no_grad():
                for _ in range(5):
                    model(x_w)
            torch.cuda.synchronize()
        except Exception as exc:
            log(f"  Warmup ERROR: {exc}")

        for b in batches:
            cl = (key != 'cudnn_nchw')
            x = make_input(b, device, channels_last=cl)
            log(f"  Batch={b} ...")
            try:
                lat, thr = benchmark_cuda_events(model, x, args.num_runs)
                log(f"    latency={lat:.3f} ms   throughput={thr:.1f} img/s")
                rows.append({'method': key, 'label': label, 'batch': b,
                             'latency_ms': f'{lat:.3f}',
                             'throughput_img_s': f'{thr:.1f}'})
            except Exception as exc:
                log(f"    ERROR: {exc}")
                rows.append({'method': key, 'label': label, 'batch': b,
                             'latency_ms': 'ERROR', 'throughput_img_s': 'ERROR'})

        del model
        torch.cuda.empty_cache()

    # Save CSV
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['method', 'label', 'batch',
                                                    'latency_ms', 'throughput_img_s'])
            writer.writeheader()
            writer.writerows(rows)
        log(f"\nResults saved to {csv_path}")

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
