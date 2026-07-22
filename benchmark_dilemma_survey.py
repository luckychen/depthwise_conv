"""E6: Dilemma survey beyond depthwise conv (re_submit.md Q2/E6).

Tests whether the three-horn integration dilemma (compile-lowering degradation /
opaque-wrapping fusion loss / layout-dependent fallback) generalizes to other
memory-bound, layout-sensitive ops:
  (a) grouped conv, groups=C/8 and C/16, at C3 and C5 shapes
  (b) dilated large-K depthwise conv (K=13, dilation=3, C3 shape)
  (c) 3D depthwise conv (3x7x7, C=384, T=8, 28x28)

For each op: eager-NCHW, eager-NHWC(channels_last), compile-default(NCHW),
opaque-wrapped-under-compile(NCHW). B=1, fp16, warm 500/500 CUDA events.
Dispatch identity captured via torch.profiler kernel names.
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

WARMUP = 500
TIMED = 500
device = "cuda"

# ---- opaque wrapping harness (pattern from benchmark_cudnn_opaque.py) ----

_lib2d = torch.library.Library("dilemma_survey_2d", "DEF")
_lib2d.define("conv2d_op(Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, int groups) -> Tensor")

@torch.library.impl(_lib2d, "conv2d_op", "CUDA")
def _conv2d_op_impl(input, weight, bias, stride, padding, dilation, groups):
    return F.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups).contiguous()

@torch.library.impl(_lib2d, "conv2d_op", "Meta")
def _conv2d_op_meta(input, weight, bias, stride, padding, dilation, groups):
    return F.conv2d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


_lib3d = torch.library.Library("dilemma_survey_3d", "DEF")
_lib3d.define("conv3d_op(Tensor input, Tensor weight, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation, int groups) -> Tensor")

@torch.library.impl(_lib3d, "conv3d_op", "CUDA")
def _conv3d_op_impl(input, weight, bias, stride, padding, dilation, groups):
    return F.conv3d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups).contiguous()

@torch.library.impl(_lib3d, "conv3d_op", "Meta")
def _conv3d_op_meta(input, weight, bias, stride, padding, dilation, groups):
    return F.conv3d(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


class Opaque2d(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride = list(conv.stride)
        self.padding = list(conv.padding) if not isinstance(conv.padding, str) else [conv.kernel_size[0] // 2] * 2
        self.dilation = list(conv.dilation)
        self.groups = conv.groups

    def forward(self, x):
        return torch.ops.dilemma_survey_2d.conv2d_op(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class Opaque3d(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride = list(conv.stride)
        self.padding = list(conv.padding)
        self.dilation = list(conv.dilation)
        self.groups = conv.groups

    def forward(self, x):
        return torch.ops.dilemma_survey_3d.conv3d_op(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


# ---- timing + dispatch identity ----

def time_and_identify(fn, x, warmup=WARMUP, timed=TIMED):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timed):
            fn(x)
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) / timed * 1000

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                fn(x)
            torch.cuda.synchronize()
    events = [e for e in prof.key_averages()
              if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0]
    events.sort(key=lambda e: -e.self_device_time_total)
    top_kernel = events[0].key if events else "?"
    dispatch = classify(top_kernel)
    return us, top_kernel, dispatch


def classify(kernel_name):
    k = kernel_name.lower()
    if "cudnn" in k or "xmma" in k or "implicit_gemm" in k:
        return "cuDNN"
    if "triton" in k:
        return "Triton"
    if "at::native" in k or "conv_depthwise" in k:
        return "ATen"
    return "mixed/other"


# ---- ops ----

def build_grouped_conv(C, K, HW, groups, device):
    conv = nn.Conv2d(C, C, K, groups=groups, padding=K // 2, bias=False).to(device).half()
    x = torch.randn(1, C, HW, HW, device=device, dtype=torch.float16)
    return conv, x


def build_dilated_dw(C, K, HW, dilation, device):
    pad = (K - 1) * dilation // 2
    conv = nn.Conv2d(C, C, K, groups=C, padding=pad, dilation=dilation, bias=False).to(device).half()
    x = torch.randn(1, C, HW, HW, device=device, dtype=torch.float16)
    return conv, x


def build_3d_dw(C, T, HW, device):
    conv = nn.Conv3d(C, C, (3, 7, 7), groups=C, padding=(1, 3, 3), bias=False).to(device).half()
    x = torch.randn(1, C, T, HW, HW, device=device, dtype=torch.float16)
    return conv, x


OPS = [
    ("grouped_C3_g8", lambda d: build_grouped_conv(128, 31, 56, 128 // 8, d)),
    ("grouped_C3_g16", lambda d: build_grouped_conv(128, 31, 56, 128 // 16, d)),
    ("grouped_C5_g8", lambda d: build_grouped_conv(512, 27, 14, 512 // 8, d)),
    ("grouped_C5_g16", lambda d: build_grouped_conv(512, 27, 14, 512 // 16, d)),
    ("dilated_dw_k13_d3_C3shape", lambda d: build_dilated_dw(128, 13, 56, 3, d)),
    ("conv3d_dw_3x7x7_C384_T8", lambda d: build_3d_dw(384, 8, 28, d)),
]


def bench_op(name, builder, device, is_3d):
    rows = []
    conv_nchw, x_nchw = builder(device)

    # eager NCHW
    us, kernel, dispatch = time_and_identify(lambda inp: conv_nchw(inp), x_nchw)
    rows.append(dict(op=name, variant="eager-NCHW", latency_us=f"{us:.3f}", kernel_name=kernel, dispatch=dispatch))
    eager_nchw_us = us

    # eager NHWC (channels_last / channels_last_3d) -- use an INDEPENDENT copy;
    # nn.Module.to(memory_format=...) mutates in place and returns self, so reusing
    # conv_nchw here would silently convert its weight for all later NCHW tests too.
    import copy
    mem_fmt = torch.channels_last_3d if is_3d else torch.channels_last
    conv_nhwc = copy.deepcopy(conv_nchw).to(memory_format=mem_fmt)
    x_nhwc = x_nchw.clone().to(memory_format=mem_fmt)
    assert conv_nchw.weight.is_contiguous(), "contamination check: NCHW weight must stay contiguous"
    try:
        us, kernel, dispatch = time_and_identify(lambda inp: conv_nhwc(inp), x_nhwc)
        rows.append(dict(op=name, variant="eager-NHWC", latency_us=f"{us:.3f}", kernel_name=kernel, dispatch=dispatch))
    except Exception as e:
        rows.append(dict(op=name, variant="eager-NHWC", latency_us="", kernel_name=f"FAILED: {e}", dispatch=""))

    # compile-default (NCHW)
    torch._dynamo.reset()
    compiled = torch.compile(conv_nchw, mode="default")
    try:
        us, kernel, dispatch = time_and_identify(lambda inp: compiled(inp), x_nchw)
        rows.append(dict(op=name, variant="compile-default-NCHW", latency_us=f"{us:.3f}", kernel_name=kernel, dispatch=dispatch))
        compile_us = us
    except Exception as e:
        rows.append(dict(op=name, variant="compile-default-NCHW", latency_us="", kernel_name=f"FAILED: {e}", dispatch=""))
        compile_us = None
    torch._dynamo.reset()

    # opaque-wrapped under compile (NCHW)
    opaque_mod = (Opaque3d(conv_nchw) if is_3d else Opaque2d(conv_nchw)).to(device)
    compiled_opaque = torch.compile(opaque_mod, mode="default")
    try:
        us, kernel, dispatch = time_and_identify(lambda inp: compiled_opaque(inp), x_nchw)
        rows.append(dict(op=name, variant="opaque-under-compile-NCHW", latency_us=f"{us:.3f}", kernel_name=kernel, dispatch=dispatch))
        opaque_us = us
    except Exception as e:
        rows.append(dict(op=name, variant="opaque-under-compile-NCHW", latency_us="", kernel_name=f"FAILED: {e}", dispatch=""))
        opaque_us = None
    torch._dynamo.reset()

    # horn classification
    compile_degradation = (compile_us / eager_nchw_us) if compile_us else None
    opaque_vs_compile = (opaque_us / compile_us) if (opaque_us and compile_us) else None
    print(f"=== {name} ===")
    for r in rows:
        print(f"  {r['variant']:<28} {r['latency_us']:>10} us  kernel={r['kernel_name'][:50]:<50} dispatch={r['dispatch']}")
    print(f"  compile/eager ratio: {compile_degradation}")
    print(f"  opaque/compile ratio: {opaque_vs_compile}")
    return rows, compile_degradation, opaque_vs_compile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    all_rows = []
    summary = []
    for name, builder in OPS:
        is_3d = "conv3d" in name
        rows, compile_deg, opaque_ratio = bench_op(name, lambda d: builder(d), device, is_3d)
        all_rows.extend(rows)
        summary.append((name, compile_deg, opaque_ratio))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["op", "variant", "latency_us", "kernel_name", "dispatch"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {args.output}")

    print("\n=== Horn summary ===")
    print(f"{'op':<26} {'compile/eager (horn1)':>22} {'opaque/compile (horn2)':>22}")
    for name, cd, orr in summary:
        cd_s = f"{cd:.2f}x" if cd else "n/a"
        or_s = f"{orr:.2f}x" if orr else "n/a"
        print(f"{name:<26} {cd_s:>22} {or_s:>22}")


if __name__ == "__main__":
    main()
