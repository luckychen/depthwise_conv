#!/usr/bin/env python3
"""
Single-kernel profiling harness for NCU.

Exercises ONE kernel configuration per invocation so NCU can isolate metrics.
Usage:
    python profile_ncu.py --config 9x9_1536_32 --method custom
    python profile_ncu.py --config 31x31_128_56 --method cudnn_nchw

Methods: custom, cudnn_nchw, cudnn_nhwc, triton_nhwc
Configs: 9x9_1536_32, 7x7_768_32, 31x31_128_56, 29x29_256_28, 27x27_512_14, 13x13_1024_7
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F

# --- config definitions -------------------------------------------------------

CONFIGS = {
    "9x9_1536_32":  {"C": 1536, "K": 9,  "H": 32, "W": 32},
    "7x7_768_32":   {"C": 768,  "K": 7,  "H": 32, "W": 32},
    "31x31_128_56": {"C": 128,  "K": 31, "H": 56, "W": 56},
    "29x29_256_28": {"C": 256,  "K": 29, "H": 28, "W": 28},
    "27x27_512_14": {"C": 512,  "K": 27, "H": 14, "W": 14},
    "13x13_1024_7": {"C": 1024, "K": 13, "H": 7,  "W": 7},
}

# --- custom kernel loading ----------------------------------------------------

_custom_loaded = False
_custom_ops = None
_replknet_loaded = False
_replknet_ops = None


def _ensure_custom_convmixer():
    global _custom_loaded, _custom_ops
    if _custom_loaded:
        return _custom_ops
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "convmixer", "kernels"))
    import depthwise_torch_ops as dto
    _custom_ops = dto._load_kernels()
    _custom_loaded = True
    return _custom_ops


def _ensure_custom_replknet():
    global _replknet_loaded, _replknet_ops
    if _replknet_loaded:
        return _replknet_ops
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "replknet", "kernels"))
    import depthwise_replknet_nhwc_ops as rno
    _replknet_ops = rno._load_kernels()
    _replknet_loaded = True
    return _replknet_ops


# --- triton kernel loading ----------------------------------------------------

_triton_kernel = None


def _ensure_triton():
    global _triton_kernel
    if _triton_kernel is not None:
        return _triton_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _dw_conv_nhwc(
        input_ptr, weight_ptr, output_ptr,
        H, W, C, K, PAD,
        stride_h, stride_w,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_hw = H * W
        num_c_blocks = tl.cdiv(C, BLOCK_C)
        hw_idx = pid // num_c_blocks
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
                        w_off = kh * K * C + kw * C + c_offsets
                        w_vals = tl.load(weight_ptr + w_off, mask=c_mask, other=0.0).to(tl.float32)
                        acc += x_vals * w_vals
        out_off = oh * stride_h + ow * stride_w + c_offsets
        tl.store(output_ptr + out_off, acc.to(tl.float16), mask=c_mask)

    _triton_kernel = _dw_conv_nhwc
    return _triton_kernel


# --- kernel runners -----------------------------------------------------------

def run_custom(cfg, x, weight, output):
    C, K, H, W = cfg["C"], cfg["K"], cfg["H"], cfg["W"]
    if C == 1536 and K == 9 and H == 32:
        ops = _ensure_custom_convmixer()
        ops.depthwise_forward_32_1536_9(x, weight, output, H, W, C, K)
    elif C == 768 and K == 7 and H == 32:
        ops = _ensure_custom_convmixer()
        ops.depthwise_forward_32_768_7(x, weight, output, H, W, C, K)
    elif C == 128 and K == 31 and H == 56:
        ops = _ensure_custom_replknet()
        bias = torch.empty(0, device=x.device, dtype=torch.float16)
        ops.depthwise_31x31_128_nhwc(x, weight, bias)
    elif C == 256 and K == 29 and H == 28:
        ops = _ensure_custom_replknet()
        bias = torch.empty(0, device=x.device, dtype=torch.float16)
        ops.depthwise_29x29_256_nhwc(x, weight, bias)
    elif C == 512 and K == 27 and H == 14:
        ops = _ensure_custom_replknet()
        bias = torch.empty(0, device=x.device, dtype=torch.float16)
        ops.depthwise_27x27_512_nhwc(x, weight, bias)
    else:
        raise RuntimeError(f"No custom kernel for config {cfg}")


def run_cudnn_nchw(cfg, x_nchw, weight_nchw):
    C, K = cfg["C"], cfg["K"]
    pad = K // 2
    return F.conv2d(x_nchw, weight_nchw, padding=pad, groups=C)


def run_cudnn_nhwc(cfg, x_cl, weight_nhwc):
    C, K = cfg["C"], cfg["K"]
    pad = K // 2
    return F.conv2d(x_cl, weight_nhwc, padding=pad, groups=C)


def run_triton_nhwc(cfg, x, weight, output):
    C, K, H, W = cfg["C"], cfg["K"], cfg["H"], cfg["W"]
    pad = K // 2
    kernel_fn = _ensure_triton()
    stride_h = W * C
    stride_w = C
    BLOCK_C = 64 if C >= 512 else 32
    grid = (H * W * ((C + BLOCK_C - 1) // BLOCK_C),)
    kernel_fn[grid](
        x, weight, output,
        H, W, C, K, pad,
        stride_h, stride_w,
        BLOCK_C=BLOCK_C,
    )


# --- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NCU profiling harness for depthwise conv")
    parser.add_argument("--config", required=True,
                        choices=list(CONFIGS.keys()),
                        help="Kernel configuration to profile")
    parser.add_argument("--method", required=True,
                        choices=["custom", "cudnn_nchw", "cudnn_nhwc", "triton_nhwc"],
                        help="Implementation method")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    C, K, H, W = cfg["C"], cfg["K"], cfg["H"], cfg["W"]
    pad = K // 2
    device = torch.device(args.device)

    print(f"Config: {args.config}  Method: {args.method}")
    print(f"C={C} K={K} H={H} W={W}")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Warmup: {args.warmup}  Iterations: {args.iterations}")
    sys.stdout.flush()

    # Set up tensors based on method
    if args.method == "custom":
        is_replknet = args.config in ("31x31_128_56", "29x29_256_28", "27x27_512_14")
        if is_replknet:
            # RepLKNet kernels expect [B, C, H, W] channels_last
            x = torch.randn(1, C, H, W, device=device, dtype=torch.float16)
            x = x.to(memory_format=torch.channels_last)
            weight = torch.randn(K, K, C, device=device, dtype=torch.float16).contiguous()
            output = torch.empty(1, C, H, W, device=device, dtype=torch.float16)
            output = output.to(memory_format=torch.channels_last)
        else:
            # ConvMixer kernels expect [H, W, C] NHWC
            x = torch.randn(H, W, C, device=device, dtype=torch.float16).contiguous()
            weight = torch.randn(K, K, C, device=device, dtype=torch.float16).contiguous()
            output = torch.empty(H, W, C, device=device, dtype=torch.float16).contiguous()
        # Warmup
        for i in range(args.warmup):
            run_custom(cfg, x, weight, output)
        torch.cuda.synchronize(device)
        # Profiling loop
        for i in range(args.iterations):
            run_custom(cfg, x, weight, output)
        torch.cuda.synchronize(device)

    elif args.method == "cudnn_nchw":
        # NCHW tensors: [1, C, H, W]
        x = torch.randn(1, C, H, W, device=device, dtype=torch.float16)
        weight = torch.empty(C, 1, K, K, device=device, dtype=torch.float16)
        w_src = torch.randn(K, K, C, device=device, dtype=torch.float16)
        weight.copy_(w_src.permute(2, 0, 1).unsqueeze(1))
        for i in range(args.warmup):
            run_cudnn_nchw(cfg, x, weight)
        torch.cuda.synchronize(device)
        for i in range(args.iterations):
            run_cudnn_nchw(cfg, x, weight)
        torch.cuda.synchronize(device)

    elif args.method == "cudnn_nhwc":
        # channels_last tensors: [1, C, H, W] channels_last
        x = torch.randn(1, C, H, W, device=device, dtype=torch.float16)
        x = x.to(memory_format=torch.channels_last)
        weight_cl = torch.empty(C, 1, K, K, device=device, dtype=torch.float16)
        weight_cl = weight_cl.to(memory_format=torch.channels_last)
        w_src = torch.randn(K, K, C, device=device, dtype=torch.float16)
        weight_cl.copy_(w_src.permute(2, 0, 1).unsqueeze(1))
        for i in range(args.warmup):
            run_cudnn_nhwc(cfg, x, weight_cl)
        torch.cuda.synchronize(device)
        for i in range(args.iterations):
            run_cudnn_nhwc(cfg, x, weight_cl)
        torch.cuda.synchronize(device)

    elif args.method == "triton_nhwc":
        x = torch.randn(H, W, C, device=device, dtype=torch.float16).contiguous()
        weight = torch.randn(K, K, C, device=device, dtype=torch.float16).contiguous()
        output = torch.empty(H, W, C, device=device, dtype=torch.float16).contiguous()
        for i in range(args.warmup):
            run_triton_nhwc(cfg, x, weight, output)
        torch.cuda.synchronize(device)
        for i in range(args.iterations):
            run_triton_nhwc(cfg, x, weight, output)
        torch.cuda.synchronize(device)

    print("Done.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()