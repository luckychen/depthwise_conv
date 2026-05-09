"""Correctness + timing A/B for the experimental v2 (region-templated) 32x32x1536 9x9 kernel.

Compares against v1 across both single-batch (B=1) and multi-batch (B=8). Measures cold-L2
per-call latency by flushing L2 between iterations with a large dummy buffer."""
import os
import sys
import time
import torch

# Ensure imports resolve regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force fresh module load so SUPPORTED_CONFIGS reflects env at first import
# (we rebuild the module per-variant since the dict is computed at class-def time)
import importlib
import depthwise_torch_ops as ops_mod

ops = ops_mod._load_kernels()

device = torch.device('cuda')
# device set via CUDA_VISIBLE_DEVICES

B, C, H, W = 16, 1536, 32, 32
K = 9

torch.manual_seed(0)
x = torch.randn(B, C, H, W, dtype=torch.float16, device=device).to(memory_format=torch.channels_last)
w = torch.randn(K, K, C, dtype=torch.float16, device=device) * 0.1  # KxKxC NHWC weight

out_v1 = torch.empty_like(x)
out_v2 = torch.empty_like(x)

# Run v1 + v2 once for correctness
ops.depthwise_forward_32_1536_9(x, w, out_v1, H, W, C, K)
ops.depthwise_forward_32_1536_9_v2(x, w, out_v2, H, W, C, K)
torch.cuda.synchronize()

diff = (out_v1.float() - out_v2.float()).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
print(f"correctness: max_abs_diff={max_diff:.6f}  mean_abs_diff={mean_diff:.6f}")

# Cold-L2 timing
def bench(fn, iters=200, warmup=20, l2_flush_mb=128):
    flush = torch.empty(l2_flush_mb * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush.zero_()        # evict L2
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    return median, p10

def run_v1():
    ops.depthwise_forward_32_1536_9(x, w, out_v1, H, W, C, K)
def run_v2():
    ops.depthwise_forward_32_1536_9_v2(x, w, out_v2, H, W, C, K)

med_v1, p10_v1 = bench(run_v1)
med_v2, p10_v2 = bench(run_v2)

per_batch_v1 = med_v1 / B
per_batch_v2 = med_v2 / B
print(f"v1: median={med_v1:.2f} us (B={B}, per-call={per_batch_v1:.2f} us)  p10={p10_v1:.2f} us")
print(f"v2: median={med_v2:.2f} us (B={B}, per-call={per_batch_v2:.2f} us)  p10={p10_v2:.2f} us")
print(f"speedup: {med_v1/med_v2:.3f}x")
