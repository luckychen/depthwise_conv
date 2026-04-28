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
import triton
import triton.language as tl

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPLKNET_REPO = os.path.join(_THIS_DIR, 'RepLKNet-pytorch')
sys.path.insert(0, REPLKNET_REPO)
sys.path.insert(0, os.path.join(_THIS_DIR, 'kernels'))

CHECKPOINT = os.path.join(_THIS_DIR, 'RepLKNet-31B_ImageNet-1K_224.pth')


# ---------------------------------------------------------------------------
# Triton depthwise conv — TWO native kernels, zero layout conversion each.
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
    """NHWC-native Triton: reads channels_last memory directly, zero conversion."""
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
    """NCHW-native Triton: reads contiguous memory directly, zero conversion."""
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

def replace_depthwise_with_triton_nhwc(model, verbose=True):
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            if module.groups == module.in_channels == module.out_channels:
                channels = module.in_channels
                kernel_size = module.kernel_size[0]
                triton_conv = TritonDepthwiseNHWC(
                    channels, kernel_size, bias=(module.bias is not None)
                ).to(module.weight.device)
                with torch.no_grad():
                    w = module.weight.squeeze(1).permute(1, 2, 0).contiguous()
                    triton_conv.weight.copy_(w.to(torch.float16))
                    if module.bias is not None:
                        triton_conv.bias.copy_(module.bias.to(torch.float16))
                parts = name.split('.')
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], triton_conv)
                replaced += 1
                if verbose:
                    print(f"  [Triton-NHWC] {name}: {channels}ch, {kernel_size}x{kernel_size}")
    if verbose:
        print(f"Replaced {replaced} depthwise conv(s) with Triton NHWC-native kernels")
    return model


def replace_depthwise_with_triton_nchw(model, verbose=True):
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            if module.groups == module.in_channels == module.out_channels:
                channels = module.in_channels
                kernel_size = module.kernel_size[0]
                triton_conv = TritonDepthwiseNCHW(
                    channels, kernel_size, bias=(module.bias is not None)
                ).to(module.weight.device)
                with torch.no_grad():
                    triton_conv.weight.copy_(module.weight.to(torch.float16))
                    if module.bias is not None:
                        triton_conv.bias.copy_(module.bias.to(torch.float16))
                parts = name.split('.')
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], triton_conv)
                replaced += 1
                if verbose:
                    print(f"  [Triton-NCHW] {name}: {channels}ch, {kernel_size}x{kernel_size}")
    if verbose:
        print(f"Replaced {replaced} depthwise conv(s) with Triton NCHW-native kernels")
    return model


def replace_depthwise_with_triton(model, verbose=True):
    """Default: NHWC native (for channels_last models)."""
    return replace_depthwise_with_triton_nhwc(model, verbose=verbose)


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
    parser.add_argument('--skip_accuracy', action='store_true',
                        help='Skip ImageNet accuracy validation')
    parser.add_argument('--data_dir', default='./data/imagenet',
                        help='ImageNet data directory (HuggingFace cache)')
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

    # Config 5: cuDNN NCHW-native eager (model in contiguous NCHW, zero conversion)
    print("\n" + "=" * 70)
    print("Config 5: CuDNN NCHW-native (eager)")
    print("=" * 70)
    model = load_model(device)
    model = model.to(memory_format=torch.contiguous_format)
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    mean, std, mn, mx = benchmark(model, x, device, args.num_runs, args.warmup)
    results['CuDNN NCHW eager'] = mean
    print(f"  {mean:.2f} +/- {std:.2f} ms  (min {mn:.2f}, max {mx:.2f})")
    if args.per_layer:
        dw_totals['CuDNN NCHW eager'] = profile_breakdown(model, x, device, 'CuDNN NCHW eager')
    del model, x
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    # Config 6: cuDNN NCHW-native + compile (zero conversion)
    print("\n" + "=" * 70)
    print("Config 6: CuDNN NCHW-native + torch.compile")
    print("=" * 70)
    model = load_model(device)
    model = model.to(memory_format=torch.contiguous_format)
    model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    mean, std, mn, mx = benchmark(model, x, device, args.num_runs, args.warmup)
    results['CuDNN NCHW+compile'] = mean
    print(f"  {mean:.2f} +/- {std:.2f} ms  (min {mn:.2f}, max {mx:.2f})")
    del model, x
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    # Config 7: Triton NHWC-native + compile (model stays channels_last)
    print("\n" + "=" * 70)
    print("Config 7: Triton NHWC-native + torch.compile")
    print("=" * 70)
    model = load_model(device)
    model = replace_depthwise_with_triton_nhwc(model, verbose=True)
    model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
    x = make_input(device)
    mean, std, mn, mx = benchmark(model, x, device, args.num_runs, args.warmup)
    results['Triton NHWC+compile'] = mean
    print(f"  {mean:.2f} +/- {std:.2f} ms  (min {mn:.2f}, max {mx:.2f})")
    del model, x
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    # Config 8: Triton NCHW-native + compile (model in contiguous NCHW)
    print("\n" + "=" * 70)
    print("Config 8: Triton NCHW-native + torch.compile")
    print("=" * 70)
    model = load_model(device)
    model = model.to(memory_format=torch.contiguous_format)
    model = replace_depthwise_with_triton_nchw(model, verbose=True)
    model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    mean, std, mn, mx = benchmark(model, x, device, args.num_runs, args.warmup)
    results['Triton NCHW+compile'] = mean
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

    # Accuracy validation
    if not args.skip_accuracy:
        print("\n" + "=" * 70)
        print("ACCURACY VALIDATION (full 50K ImageNet-1K)")
        print("=" * 70)
        try:
            saved_path = sys.path[:]
            sys.path = [p for p in sys.path if 'RepLKNet-pytorch' not in p]
            from datasets import load_dataset
            sys.path = saved_path
            from torchvision import transforms
            from pathlib import Path

            cache_path = Path(args.data_dir)
            if cache_path.exists():
                print(f"Loading ImageNet from cache: {args.data_dir}")
                dataset = load_dataset(
                    'ILSVRC/imagenet-1k',
                    split='validation',
                    cache_dir=args.data_dir,
                    verification_mode='no_checks'
                )
            else:
                print("Downloading ImageNet from HuggingFace...")
                dataset = load_dataset(
                    'ILSVRC/imagenet-1k',
                    split='validation',
                    cache_dir=args.data_dir
                )

            val_transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            class HFImageNetDataset(torch.utils.data.Dataset):
                def __init__(self, hf_dataset):
                    self.dataset = hf_dataset

                def __len__(self):
                    return len(self.dataset)

                def __getitem__(self, idx):
                    item = self.dataset[idx]
                    img = item['image'].convert('RGB')
                    return val_transform(img), item['label']

            val_dataset = HFImageNetDataset(dataset)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False,
                num_workers=4, pin_memory=True
            )

            model = load_model(device)
            model = replace_replknet_depthwise_nhwc(model, verbose=False)
            model = torch.compile(model, mode='reduce-overhead', fullgraph=False)

            print(f"\nValidating RepLKNet-31B (custom+compile)...")
            correct1, correct5, total = 0, 0, 0
            t0 = time.time()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, dtype=torch.float16,
                                       memory_format=torch.channels_last)
                    labels = labels.to(device)
                    out = model(images)
                    _, pred = out.topk(5, dim=1)
                    correct1 += (pred[:, 0] == labels).sum().item()
                    correct5 += sum(labels[i].item() in pred[i].tolist()
                                    for i in range(labels.size(0)))
                    total += labels.size(0)
                    if total % 1000 == 0:
                        elapsed = time.time() - t0
                        print(f"  [{total:>6d}/{len(val_dataset)}] "
                              f"Top-1: {100*correct1/total:.2f}%  "
                              f"Top-5: {100*correct5/total:.2f}%  "
                              f"({elapsed:.0f}s, {total/elapsed:.0f} img/s)")

            elapsed = time.time() - t0
            top1 = 100 * correct1 / total
            top5 = 100 * correct5 / total
            print(f"  FINAL: Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")

            del model
            torch.cuda.empty_cache()

        except ImportError:
            print("  Skipped: 'datasets' package not installed (pip install datasets)")
        except Exception as e:
            print(f"  Skipped: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (all FP16, native layouts, zero conversion)")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Latency':>10} {'vs CuDNN-best':>14}", end='')
    if dw_totals:
        print(f" {'DW total':>10} {'Non-DW':>10}")
    else:
        print()
    print("-" * (70 if dw_totals else 61))
    cudnn_nhwc = results.get('CuDNN eager', 999)
    cudnn_nchw = results.get('CuDNN NCHW eager', 999)
    cudnn_best = min(cudnn_nhwc, cudnn_nchw)
    for name in ['CuDNN eager', 'CuDNN NCHW eager',
                  'CuDNN+compile', 'CuDNN NCHW+compile',
                  'Triton NHWC+compile', 'Triton NCHW+compile',
                  'custom eager', 'custom+compile']:
        if name in results:
            t = results[name]
            speedup = cudnn_best / t
            line = f"{name:<35} {t:>8.2f} ms {speedup:>12.2f}x"
            if name in dw_totals:
                dw = dw_totals[name]
                non_dw = t - dw
                line += f" {dw:>8.2f} ms {non_dw:>8.2f} ms"
            print(line)
    print("=" * 70)


if __name__ == '__main__':
    main()
