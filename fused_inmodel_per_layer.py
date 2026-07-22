"""Q1.6: in-model vs standalone timing for the fused cudnn_fused::dw_act_bn op.

Same methodology as benchmark_replknet_nchw_per_layer.py (§13.10): capture each
fused DW+act+BN block's real input via forward_pre_hook during one full model
forward pass, then time each captured block in isolation (25th-percentile,
matching the established protocol). Compares against the standalone fused
per-kernel numbers already measured (E5/E5b/E5c, C2 shape: Ada 54.9us,
GB10 178.1us) to get the in-model inflation factor for cuDNN's fused engine.

Usage: python fused_inmodel_per_layer.py --output results/{ada,spark}_resub_fused_inmodel_per_layer.txt
"""
import argparse
import os
import sys
import time

import torch

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "convmixer"))

import benchmark_cudnn_fused_e2e as fe2e


def time_layer_with_captured_input(mod, inp, n_warmup=20, n_runs=100):
    with torch.no_grad():
        for _ in range(n_warmup):
            mod(inp)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mod(inp)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        return times[n_runs // 4]  # 25th percentile, matches §13.10 protocol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = "cuda"

    print(f"torch {torch.__version__}  cudnn {torch.backends.cudnn.version()}")

    m_ref, m_fused = fe2e.build_model("1536_20", 0, device)  # act_kind=0 -> gelu
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)

    # verify correctness before trusting anything (per-session discipline)
    with torch.no_grad():
        y_ref = m_ref(x)
        y_fused = m_fused(x)
    diff = (y_ref.float() - y_fused.float()).abs()
    rel = diff.max().item() / (y_ref.float().abs().mean().item() + 1e-6)
    print(f"correctness check: max abs diff={diff.max().item():.5f}  rel={rel:.5f}")

    # find all FusedDWActBN modules
    fused_blocks = {}
    for name, mod in m_fused.named_modules():
        if isinstance(mod, fe2e.FusedDWActBN):
            fused_blocks[name] = mod
    print(f"Found {len(fused_blocks)} fused DW+act+BN blocks")

    # capture real inputs via forward_pre_hook during one full forward
    activations = {}
    handles = []

    def make_hook(n):
        def hook(mod, inp):
            activations[n] = inp[0].clone()
        return hook

    for name, mod in fused_blocks.items():
        handles.append(mod.register_forward_pre_hook(make_hook(name)))

    with torch.no_grad():
        m_fused(x)
        torch.cuda.synchronize()

    for h in handles:
        h.remove()

    # time each block in isolation with its captured real input
    per_layer_ms = {}
    for name, mod in fused_blocks.items():
        inp = activations[name]
        per_layer_ms[name] = time_layer_with_captured_input(mod, inp)

    total_ms = sum(per_layer_ms.values())
    mean_us = (total_ms / len(per_layer_ms)) * 1000

    lines = []
    lines.append(f"torch {torch.__version__}  cudnn {torch.backends.cudnn.version()}")
    lines.append(f"correctness: max abs diff={diff.max().item():.5f}  rel={rel:.5f}")
    lines.append(f"n_blocks={len(fused_blocks)}")
    lines.append(f"{'block':<40} {'in-model (ms)':>14}")
    for name, ms in per_layer_ms.items():
        lines.append(f"{name:<40} {ms:>14.4f}")
    lines.append(f"\nTotal in-model fused DW+act+BN: {total_ms:.3f} ms")
    lines.append(f"Mean per-layer (in-model): {mean_us:.1f} us")

    out = "\n".join(lines)
    print(out)
    with open(args.output, "w") as f:
        f.write(out + "\n")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
