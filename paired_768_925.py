"""WQ9 step 1: paired, interleaved re-measure of 768/32-GB10 fused-cuDNN-9.25 vs Custom.

Alternates A (fused-opaque r-o) and B (Custom r-o) three times each within one
process/session to remove the ~10% run-to-run thermal/clock scatter identified
in E5d (results.md §2.9e) as a confound on cross-session comparisons.
Must run under the cuDNN 9.25 LD_PRELOAD; verifies version before trusting anything.
"""
import csv
import os
import sys

import torch

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "convmixer"))

import benchmark_cudnn_fused_e2e as fe2e

device = "cuda"
assert torch.backends.cudnn.version() == 92500, f"expected cuDNN 92500, got {torch.backends.cudnn.version()}"

# --- Build fused-opaque model (768/32, act=relu) ---
m_ref, m_fused = fe2e.build_model("768_32", 1, device)
x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
with torch.no_grad():
    y_ref = m_ref(x)
    y_fused = m_fused(x)
rel = (y_ref.float() - y_fused.float()).abs().max().item() / (y_ref.float().abs().mean().item() + 1e-6)
print(f"fused correctness check: rel={rel:.5f}")
compiled_fused = torch.compile(m_fused, mode="reduce-overhead")

# --- Build Custom model (768/32) ---
sys.path.insert(0, os.path.join(_REPO, "convmixer", "kernels"))
from convmixer import load_pretrained
from depthwise_torch_ops import replace_depthwise_convs

m_custom = load_pretrained("768_32", checkpoints_dir=os.path.join(_REPO, "convmixer", "checkpoints"),
                           device=device, channels_last=True)
m_custom = replace_depthwise_convs(m_custom, verbose=False)
m_custom = m_custom.to(dtype=torch.float16, memory_format=torch.channels_last).eval()
compiled_custom = torch.compile(m_custom, mode="reduce-overhead")

WARMUP = 500
TIMED = 500


def time_model_ms(fn, x):
    with torch.no_grad():
        for _ in range(WARMUP):
            fn(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED):
            fn(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED  # ms/call (elapsed_time is already in ms)


rows = []
for rep in range(3):
    a_ms = time_model_ms(lambda inp: compiled_fused(inp), x)
    b_ms = time_model_ms(lambda inp: compiled_custom(inp), x)
    ratio = a_ms / b_ms
    print(f"rep {rep}: fused(A)={a_ms:.4f} ms  custom(B)={b_ms:.4f} ms  ratio={ratio:.4f}")
    rows.append(dict(rep=rep, fused_ms=f"{a_ms:.4f}", custom_ms=f"{b_ms:.4f}", ratio=f"{ratio:.4f}"))

ratios = [float(r["ratio"]) for r in rows]
ratios_sorted = sorted(ratios)
median = ratios_sorted[1]
print(f"\nmedian paired ratio: {median:.4f}")

with open("results/spark_resub_paired_768_925.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["rep", "fused_ms", "custom_ms", "ratio"])
    writer.writeheader()
    writer.writerows(rows)
    f.write(f"\n# median paired ratio: {median:.4f}\n")
print("Wrote results/spark_resub_paired_768_925.csv")
