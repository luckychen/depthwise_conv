#!/usr/bin/env python3
"""Test hypothesis: torch.compile fuses custom ops better than cuDNN kernels.

Custom ops registered via torch.library give inductor a known operator it can
potentially fuse, while cuDNN internal kernels are opaque to the compiler.
"""

import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "replknet", "kernels"))
import depthwise_replknet_nhwc_ops as rno
ops = rno._load_kernels()

device = torch.device("cuda:0")
NR = 200

# Configure: 31x31, 128ch, 56x56
C, K, H = 128, 31, 56
PW_C = 256

# --- Tensors ---
# Custom NHWC: uses channels_last [1,C,H,W] for input, [K,K,C] for weight
x_nhwc = torch.randn(1, C, H, H, dtype=torch.float16, device=device).to(memory_format=torch.channels_last)
dw_w_custom = torch.randn(K, K, C, dtype=torch.float16, device=device)
dw_b_custom = torch.randn(C, dtype=torch.float16, device=device)
pw_w_nhwc = torch.randn(PW_C, C, 1, 1, dtype=torch.float16, device=device).to(memory_format=torch.channels_last)
pw_b = torch.randn(PW_C, dtype=torch.float16, device=device)

# cuDNN NCHW: contiguous [1,C,H,W], weight [C,1,K,K]
x_nchw = torch.randn(1, C, H, H, dtype=torch.float16, device=device)
dw_w_cudnn = torch.randn(C, 1, K, K, dtype=torch.float16, device=device)
pw_w_nchw = torch.randn(PW_C, C, 1, 1, dtype=torch.float16, device=device)

# cuDNN NHWC: channels_last, weight also channels_last [C,1,K,K]
x_nhwc2 = torch.randn(1, C, H, H, dtype=torch.float16, device=device).to(memory_format=torch.channels_last)
dw_w_cudnn_cl = torch.randn(C, 1, K, K, dtype=torch.float16, device=device).to(memory_format=torch.channels_last)
pw_w_nhwc2 = torch.randn(PW_C, C, 1, 1, dtype=torch.float16, device=device).to(memory_format=torch.channels_last)

# --- Blocks ---
def custom_block(x, w_dw, b_dw, w_pw, b_pw):
    x = torch.ops.replknet_dw.conv_56_128_31(x, w_dw, b_dw)
    x = F.conv2d(x, w_pw, b_pw)
    x = F.gelu(x)
    return x

def cudnn_nchw_block(x, w_dw, w_pw, b_pw):
    x = F.conv2d(x, w_dw, padding=15, groups=C)
    x = F.conv2d(x, w_pw, b_pw)
    x = F.gelu(x)
    return x

def cudnn_nhwc_block(x, w_dw, w_pw, b_pw):
    x = F.conv2d(x, w_dw, padding=15, groups=C)
    x = F.conv2d(x, w_pw, b_pw)
    x = F.gelu(x)
    return x

# --- Compile ---
print("Compiling (this may take a minute)...")
custom_comp = torch.compile(custom_block, fullgraph=False, dynamic=False)
cudnn_nchw_comp = torch.compile(cudnn_nchw_block, fullgraph=False, dynamic=False)
cudnn_nhwc_comp = torch.compile(cudnn_nhwc_block, fullgraph=False, dynamic=False)

# Warmup
print("Warming up...")
for _ in range(20):
    custom_comp(x_nhwc, dw_w_custom, dw_b_custom, pw_w_nhwc, pw_b)
    cudnn_nchw_comp(x_nchw, dw_w_cudnn, pw_w_nchw, pw_b)
    cudnn_nhwc_comp(x_nhwc2, dw_w_cudnn_cl, pw_w_nhwc2, pw_b)
torch.cuda.synchronize()

# --- Benchmark ---
def bench(fn, *args):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(NR):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / NR

custom_eager = bench(custom_block, x_nhwc, dw_w_custom, dw_b_custom, pw_w_nhwc, pw_b)
cudnn_nchw_eager = bench(cudnn_nchw_block, x_nchw, dw_w_cudnn, pw_w_nchw, pw_b)
cudnn_nhwc_eager = bench(cudnn_nhwc_block, x_nhwc2, dw_w_cudnn_cl, pw_w_nhwc2, pw_b)

custom_compiled = bench(custom_comp, x_nhwc, dw_w_custom, dw_b_custom, pw_w_nhwc, pw_b)
cudnn_nchw_compiled = bench(cudnn_nchw_comp, x_nchw, dw_w_cudnn, pw_w_nchw, pw_b)
cudnn_nhwc_compiled = bench(cudnn_nhwc_comp, x_nhwc2, dw_w_cudnn_cl, pw_w_nhwc2, pw_b)

print(f"\n{'='*75}")
print(f"Mini-block: 31x31 DW + 1x1 PW + GELU  (B=1, FP16, {NR} runs)")
print(f"{'='*75}")
print(f"{'Method':<22} {'Layout':>8} {'Eager (μs)':>12} {'Compiled (μs)':>14} {'Speedup':>8}  {'DW+PW ratio':>12}")
print(f"{'-'*22} {'-'*8} {'-'*12} {'-'*14} {'-'*8}  {'-'*12}")

# DW-only measurements for decomposition
def dw_only_custom(x, w, b):
    return torch.ops.replknet_dw.conv_56_128_31(x, w, b)

def dw_only_cudnn_nchw(x, w):
    return F.conv2d(x, w, padding=15, groups=C)

def dw_only_cudnn_nhwc(x, w):
    return F.conv2d(x, w, padding=15, groups=C)

dw_custom = bench(dw_only_custom, x_nhwc, dw_w_custom, dw_b_custom)
dw_cudnn_nchw = bench(dw_only_cudnn_nchw, x_nchw, dw_w_cudnn)
dw_cudnn_nhwc = bench(dw_only_cudnn_nhwc, x_nhwc2, dw_w_cudnn_cl)

# PW-only
def pw_only_nhwc(x, w, b):
    return F.conv2d(x, w, b)

def pw_only_nchw(x, w, b):
    return F.conv2d(x, w, b)

pw_custom_pw = bench(pw_only_nhwc, x_nhwc, pw_w_nhwc, pw_b)
pw_cudnn_nchw_pw = bench(pw_only_nchw, x_nchw, pw_w_nchw, pw_b)
pw_cudnn_nhwc_pw = bench(pw_only_nhwc, x_nhwc2, pw_w_nhwc2, pw_b)

for label, layout, eager, comp, dw_t, pw_t, pw_nhwc_t in [
    ("Custom NHWC", "NHWC", custom_eager, custom_compiled, dw_custom, custom_eager - dw_custom, pw_custom_pw),
    ("cuDNN NCHW", "NCHW", cudnn_nchw_eager, cudnn_nchw_compiled, dw_cudnn_nchw, cudnn_nchw_eager - dw_cudnn_nchw, pw_cudnn_nchw_pw),
    ("cuDNN NHWC", "NHWC", cudnn_nhwc_eager, cudnn_nhwc_compiled, dw_cudnn_nhwc, cudnn_nhwc_eager - dw_cudnn_nhwc, pw_cudnn_nhwc_pw),
]:
    sp = eager/comp
    dw_pct = (dw_t / eager) * 100
    leftover = eager - dw_t
    print(f"{label:<22} {layout:>8} {eager*1000:>12.1f} {comp*1000:>14.1f} {sp:>8.2f}x  DW={dw_pct:.0f}%")
    # Fusion benefit: how much did non-DW time shrink?
    non_dw_savings = (leftover - (comp - dw_t)) if comp > dw_t else 0
    if non_dw_savings > 0:
        print(f"  {'':>22} {'':>8} {'DW alone:':>12} {dw_t*1000:>14.1f} μs")
        print(f"  {'':>22} {'':>8} {'non-DW (PW+GELU+overhead):':>12} {leftover*1000:>14.1f} μs -> {(comp-dw_t)*1000:.1f} μs compiled (saved {(non_dw_savings/leftover)*100:.0f}%)")

# Now the key test: does inductor see FUSION opportunities?
print(f"\n{'='*75}")
print(f"Fusion check: does inductor combine DW+PW+GELU into fewer kernels?")
print(f"{'='*75}")

# CUDA events profiling - create a single-kernel trace
torch.cuda.synchronize()
# Record just one run
s = torch.cuda.Event(enable_timing=False)
e = torch.cuda.Event(enable_timing=False)

# Custom compiled
s.record()
custom_comp(x_nhwc, dw_w_custom, dw_b_custom, pw_w_nhwc, pw_b)
e.record()
torch.cuda.synchronize()

# cuDNN NCHW compiled
s2 = torch.cuda.Event(enable_timing=False)
e2 = torch.cuda.Event(enable_timing=False)
s2.record()
cudnn_nchw_comp(x_nchw, dw_w_cudnn, pw_w_nchw, pw_b)
e2.record()
torch.cuda.synchronize()

# The real way to check is NSYS.
# Let's also check: does inductor generate fused triton kernels?
print("\nChecking if inductor generates fused kernels...")
print("(Run with TORCHINDUCTOR_COMPILE_THREADS=1 and check NSYS timeline for answer)")

# Quick alternative: check the inductor output code
try:
    with open("/tmp/inductor_check.py", "w") as f:
        f.write("""
import torch
import os,sys
sys.path.insert(0, "{0}")
import depthwise_replknet_nhwc_ops as rno
ops = rno._load_kernels()

C, K, H = 128, 31, 56
PW_C = 256
device = "cuda:0"
x_nhwc = torch.randn(1, C, H, H, dtype=torch.float16, device=device).to(memory_format=torch.channels_last)
dw_w = torch.randn(K, K, C, dtype=torch.float16, device=device)
dw_b = torch.randn(C, dtype=torch.float16, device=device)
pw_w = torch.randn(PW_C, C, 1, 1, dtype=torch.float16, device=device).to(memory_format=torch.channels_last)
pw_b = torch.randn(PW_C, dtype=torch.float16, device=device)

@torch.compile(options={{"epilogue_fusion": True}})
def block(x, w_dw, b_dw, w_pw, b_pw):
    x = torch.ops.replknet_dw.conv_56_128_31(x, w_dw, b_dw)
    x = torch.nn.functional.conv2d(x, w_pw, b_pw)
    x = torch.nn.functional.gelu(x)
    return x

block(x_nhwc, dw_w, dw_b, pw_w, pw_b)
torch.cuda.synchronize()
print("Ran with epilogue_fusion=True")
        """.format(os.path.join(os.path.dirname(__file__), "replknet", "kernels")))
except:
    pass

print("\n→ Next: run NSYS to count kernel launches and see fusion in timeline")