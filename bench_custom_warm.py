"""E5b step 4: warm-parity Custom re-measures for C1, C2, C4 (C3/C5 already warm in §2.6)."""
import argparse
import csv
import os
import sys

import torch

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "replknet", "kernels"))
sys.path.insert(0, os.path.join(_REPO, "convmixer", "kernels"))

WARMUP = 500
TIMED = 500


def bench(module, x, device):
    module = module.half().to(device)
    x = x.to(device).half().to(memory_format=torch.channels_last)
    with torch.no_grad():
        for _ in range(WARMUP):
            y = module(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED):
            y = module(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED * 1000  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = args.device
    torch.manual_seed(0)

    from depthwise_torch_ops import CustomDepthwiseConv2d
    from depthwise_replknet_nhwc_ops import NHWCDepthwise29x29_256

    rows = []

    # C1: 7x7, 768ch, 32x32
    m = CustomDepthwiseConv2d(768, 7)
    x = torch.randn(1, 768, 32, 32)
    us = bench(m, x, device)
    print(f"C1 (K=7,C=768,32x32,B=1) Custom NHWC warm: {us:.3f} us")
    rows.append(dict(config="C1", K=7, C=768, H=32, W=32, custom_warm_us=f"{us:.3f}"))

    # C2: 9x9, 1536ch, 32x32
    m = CustomDepthwiseConv2d(1536, 9)
    x = torch.randn(1, 1536, 32, 32)
    us = bench(m, x, device)
    print(f"C2 (K=9,C=1536,32x32,B=1) Custom NHWC warm: {us:.3f} us")
    rows.append(dict(config="C2", K=9, C=1536, H=32, W=32, custom_warm_us=f"{us:.3f}"))

    # C4: 29x29, 256ch, 28x28
    m = NHWCDepthwise29x29_256()
    x = torch.randn(1, 256, 28, 28)
    us = bench(m, x, device)
    print(f"C4 (K=29,C=256,28x28,B=1) Custom NHWC warm: {us:.3f} us")
    rows.append(dict(config="C4", K=29, C=256, H=28, W=28, custom_warm_us=f"{us:.3f}"))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "K", "C", "H", "W", "custom_warm_us"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
