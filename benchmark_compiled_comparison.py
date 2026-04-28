#!/usr/bin/env python3
"""
End-to-end compiled comparison: cuDNN (best layout) vs Custom CUDA vs Triton.

All three are "compiled" approaches — apple-to-apple:
  - cuDNN + torch.compile (NCHW — cuDNN's best layout on GB10)
  - cuDNN + torch.compile (NHWC — channels_last, original paper baseline)
  - Custom CUDA + torch.compile (NHWC)
  - Triton replacing depthwise + torch.compile (NHWC)

Usage:
    python benchmark_compiled_comparison.py --model 1536_20 --device cuda:0 --num_runs 500
    python benchmark_compiled_comparison.py --model 768_32 --device cuda:0 --num_runs 500
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import time
import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'convmixer'))
sys.path.insert(0, os.path.join(_THIS_DIR, 'convmixer', 'kernels'))

from convmixer import load_pretrained
from depthwise_torch_ops import replace_depthwise_convs


# ---------------------------------------------------------------------------
# Triton depthwise conv — native NHWC and NCHW kernels (zero layout conversion)
# ---------------------------------------------------------------------------

# --- NHWC native: vectorize over C (contiguous channel access) ---
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=8, num_stages=2),
    ],
    key=["H", "W", "C", "K"],
)
@triton.jit
def depthwise_nhwc_native_kernel(
    input_ptr, weight_ptr, output_ptr,
    H, W, C, K, PAD,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_c_blocks = tl.cdiv(C, BLOCK_C)
    hw_idx = pid // num_c_blocks
    c_block = pid % num_c_blocks
    oh = hw_idx // W
    ow = hw_idx % W
    c_start = c_block * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    WC = W * C
    for kh in range(K):
        ih = oh + kh - PAD
        if ih >= 0 and ih < H:
            for kw in range(K):
                iw = ow + kw - PAD
                if iw >= 0 and iw < W:
                    in_off = ih * WC + iw * C + c_offsets
                    x_vals = tl.load(input_ptr + in_off, mask=c_mask, other=0.0).to(tl.float32)
                    w_off = kh * K * C + kw * C + c_offsets
                    w_vals = tl.load(weight_ptr + w_off, mask=c_mask, other=0.0).to(tl.float32)
                    acc += x_vals * w_vals
    out_off = oh * WC + ow * C + c_offsets
    tl.store(output_ptr + out_off, acc.to(tl.float16), mask=c_mask)


class TritonDepthwiseNHWC(nn.Module):
    """NHWC-native: reads channels_last memory directly, zero conversion."""
    def __init__(self, channels, kernel_size, bias=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(kernel_size, kernel_size, channels, dtype=torch.float16)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        B, C, H, W = x.shape
        output = torch.empty_like(x)
        K = self.kernel_size
        grid = lambda meta: (H * W * triton.cdiv(C, meta["BLOCK_C"]),)
        depthwise_nhwc_native_kernel[grid](
            x, self.weight, output,
            H, W, C, K, self.padding,
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output


# --- NCHW native: vectorize over W (contiguous spatial access) ---
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 64}, num_warps=8, num_stages=2),
    ],
    key=["C", "H", "W", "K"],
)
@triton.jit
def depthwise_nchw_native_kernel(
    input_ptr, weight_ptr, output_ptr,
    C, H, W, K, PAD,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_w_blocks = tl.cdiv(W, BLOCK_W)
    ch_oh_idx = pid // num_w_blocks
    w_block = pid % num_w_blocks
    c = ch_oh_idx // H
    oh = ch_oh_idx % H
    ow_offsets = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_mask = ow_offsets < W
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)
    HW = H * W
    base_in = c * HW
    base_w = c * K * K
    for kh in range(K):
        ih = oh + kh - PAD
        if ih >= 0 and ih < H:
            row_off = base_in + ih * W
            for kw in range(K):
                iw_offsets = ow_offsets + kw - PAD
                iw_mask = ow_mask & (iw_offsets >= 0) & (iw_offsets < W)
                x_vals = tl.load(input_ptr + row_off + iw_offsets, mask=iw_mask, other=0.0).to(tl.float32)
                w_val = tl.load(weight_ptr + base_w + kh * K + kw).to(tl.float32)
                acc += x_vals * w_val
    out_off = base_in + oh * W + ow_offsets
    tl.store(output_ptr + out_off, acc.to(tl.float16), mask=ow_mask)


class TritonDepthwiseNCHW(nn.Module):
    """NCHW-native: reads contiguous memory directly, zero conversion."""
    def __init__(self, channels, kernel_size, bias=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(channels, 1, kernel_size, kernel_size, dtype=torch.float16)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        B, C, H, W = x.shape
        output = torch.empty_like(x)
        K = self.kernel_size
        grid = lambda meta: (C * H * triton.cdiv(W, meta["BLOCK_W"]),)
        depthwise_nchw_native_kernel[grid](
            x, self.weight, output,
            C, H, W, K, self.padding,
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output


# --- Replacement functions ---

def replace_depthwise_triton_nhwc(model, verbose=True):
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            if module.groups == module.in_channels == module.out_channels:
                ch = module.in_channels
                ks = module.kernel_size[0]
                t = TritonDepthwiseNHWC(ch, ks, bias=(module.bias is not None)).to(module.weight.device)
                with torch.no_grad():
                    w = module.weight.squeeze(1).permute(1, 2, 0).contiguous()
                    t.weight.copy_(w.to(torch.float16))
                    if module.bias is not None:
                        t.bias.copy_(module.bias.to(torch.float16))
                parts = name.split('.')
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], t)
                replaced += 1
                if verbose:
                    print(f"  [Triton-NHWC] {name}: {ch}ch, {ks}x{ks}")
    if verbose:
        print(f"Replaced {replaced} depthwise conv(s) with Triton NHWC-native")
    return model


def replace_depthwise_triton_nchw(model, verbose=True):
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            if module.groups == module.in_channels == module.out_channels:
                ch = module.in_channels
                ks = module.kernel_size[0]
                t = TritonDepthwiseNCHW(ch, ks, bias=(module.bias is not None)).to(module.weight.device)
                with torch.no_grad():
                    t.weight.copy_(module.weight.to(torch.float16))
                    if module.bias is not None:
                        t.bias.copy_(module.bias.to(torch.float16))
                parts = name.split('.')
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], t)
                replaced += 1
                if verbose:
                    print(f"  [Triton-NCHW] {name}: {ch}ch, {ks}x{ks}")
    if verbose:
        print(f"Replaced {replaced} depthwise conv(s) with Triton NCHW-native")
    return model


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark_one(model, device, num_runs=500, warmup=200, memory_format=torch.channels_last, real_input=None):
    if real_input is not None:
        x = real_input.to(memory_format=memory_format)
    else:
        x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=memory_format)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = model(x)
        end.record()
        torch.cuda.synchronize(device)
        latency_ms = start.elapsed_time(end) / num_runs

    throughput = 1000.0 / latency_ms
    return latency_ms, throughput


def load_model(model_name, device):
    return load_pretrained(
        model_name,
        checkpoints_dir=os.path.join(_THIS_DIR, 'convmixer', 'checkpoints'),
        device=device, channels_last=True
    )


def main():
    parser = argparse.ArgumentParser(description='Compiled end-to-end comparison')
    parser.add_argument('--model', default='1536_20', choices=['1536_20', '768_32'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_runs', type=int, default=500)
    parser.add_argument('--data_dir', default='./data/imagenet',
                        help='ImageNet data dir (HuggingFace cache) for real input')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    print("=" * 78)
    print(f"Compiled End-to-End Comparison: ConvMixer-{args.model}")
    print("=" * 78)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}")
    print(f"Batch=1, FP16, {args.num_runs} runs")
    print()

    # Load real ImageNet image
    real_input = None
    try:
        from pathlib import Path
        from datasets import load_dataset
        from torchvision import transforms

        cache_path = Path(args.data_dir)
        if cache_path.exists():
            print("Loading real ImageNet validation image...")
            dataset = load_dataset(
                'ILSVRC/imagenet-1k',
                split='validation',
                cache_dir=args.data_dir,
                verification_mode='no_checks'
            )
            resize = 233 if '1536' in args.model else 256
            interp = (transforms.InterpolationMode.BICUBIC if '1536' in args.model
                      else transforms.InterpolationMode.BILINEAR)
            val_transform = transforms.Compose([
                transforms.Resize(resize, interpolation=interp),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            img = dataset[0]['image'].convert('RGB')
            real_input = val_transform(img).unsqueeze(0).to(device, dtype=torch.float16)
            print(f"  Using real image (label={dataset[0]['label']})\n")
        else:
            print(f"Data dir {args.data_dir} not found, using random input\n")
    except Exception as e:
        print(f"Could not load real data ({e}), using random input\n")

    results = {}

    # --- 1. cuDNN + compile (NCHW — cuDNN's best layout on GB10) ---
    print("Loading cuDNN + compile (NCHW)...")
    m = load_model(args.model, device)
    m = m.float().to(device)  # NCHW contiguous
    m = m.half()
    m.channels_last = False
    m = torch.compile(m, mode='default')
    lat, thr = benchmark_one(m, device, args.num_runs, memory_format=torch.contiguous_format, real_input=real_input)
    results['cuDNN+compile (NCHW)'] = (lat, thr)
    print(f"  {lat:.2f} ms  ({thr:.1f} img/s)")
    del m; torch.cuda.empty_cache()

    # --- 2. cuDNN + compile (NHWC) ---
    print("Loading cuDNN + compile (NHWC)...")
    m = load_model(args.model, device)
    m = m.half().to(memory_format=torch.channels_last)
    m = torch.compile(m, mode='default')
    lat, thr = benchmark_one(m, device, args.num_runs, memory_format=torch.channels_last, real_input=real_input)
    results['cuDNN+compile (NHWC)'] = (lat, thr)
    print(f"  {lat:.2f} ms  ({thr:.1f} img/s)")
    del m; torch.cuda.empty_cache()

    # --- 3. Custom CUDA + compile (NHWC) ---
    print("Loading Custom CUDA + compile (NHWC)...")
    m = load_model(args.model, device)
    m = m.half().to(memory_format=torch.channels_last)
    m = replace_depthwise_convs(m, verbose=False)
    m = torch.compile(m, mode='default')
    lat, thr = benchmark_one(m, device, args.num_runs, memory_format=torch.channels_last, real_input=real_input)
    results['Custom+compile (NHWC)'] = (lat, thr)
    print(f"  {lat:.2f} ms  ({thr:.1f} img/s)")
    del m; torch.cuda.empty_cache()

    # --- 4. Triton NHWC-native + compile ---
    print("Loading Triton NHWC-native + compile...")
    m = load_model(args.model, device)
    m = m.half().to(memory_format=torch.channels_last)
    m = replace_depthwise_triton_nhwc(m, verbose=False)
    m = torch.compile(m, mode='default')
    lat, thr = benchmark_one(m, device, args.num_runs, memory_format=torch.channels_last, real_input=real_input)
    results['Triton NHWC+compile'] = (lat, thr)
    print(f"  {lat:.2f} ms  ({thr:.1f} img/s)")
    del m; torch.cuda.empty_cache()

    # --- 5. Triton NCHW-native + compile ---
    print("Loading Triton NCHW-native + compile...")
    m = load_model(args.model, device)
    m = m.float().to(device)
    m = m.half()
    m = replace_depthwise_triton_nchw(m, verbose=False)
    m = torch.compile(m, mode='default')
    lat, thr = benchmark_one(m, device, args.num_runs, memory_format=torch.contiguous_format, real_input=real_input)
    results['Triton NCHW+compile'] = (lat, thr)
    print(f"  {lat:.2f} ms  ({thr:.1f} img/s)")
    del m; torch.cuda.empty_cache()

    # --- Summary ---
    print()
    print("=" * 78)
    print(f"SUMMARY: ConvMixer-{args.model} (batch=1, FP16, all compiled)")
    print("=" * 78)

    # Find best cuDNN
    cudnn_nchw = results['cuDNN+compile (NCHW)'][0]
    cudnn_nhwc = results['cuDNN+compile (NHWC)'][0]
    cudnn_best = min(cudnn_nchw, cudnn_nhwc)
    cudnn_best_name = 'NCHW' if cudnn_nchw < cudnn_nhwc else 'NHWC'

    print(f"\n{'Method':<30} {'Latency':>10} {'Throughput':>12} {'vs cuDNN-best':>14}")
    print("-" * 70)
    for name, (lat, thr) in results.items():
        speedup = cudnn_best / lat
        marker = " <-- cuDNN best" if lat == cudnn_best else ""
        print(f"{name:<30} {lat:>8.2f} ms {thr:>9.1f} img/s {speedup:>12.2f}x{marker}")

    # Find best Triton
    triton_results = {k: v[0] for k, v in results.items() if 'Triton' in k}
    if triton_results:
        triton_best_name = min(triton_results, key=triton_results.get)
        triton_best = triton_results[triton_best_name]
        custom_lat = results.get('Custom+compile (NHWC)', (0,))[0]
        print(f"\ncuDNN best layout: {cudnn_best_name} ({cudnn_best:.2f} ms)")
        print(f"Triton best: {triton_best_name} ({triton_best:.2f} ms)")
        if custom_lat > 0:
            print(f"Custom vs Triton-best: {triton_best/custom_lat:.2f}x faster")
    else:
        print(f"\ncuDNN best layout: {cudnn_best_name} ({cudnn_best:.2f} ms)")
    print("=" * 78)


if __name__ == '__main__':
    main()
