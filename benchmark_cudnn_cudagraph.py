"""E4: eager (no torch.compile) + manual CUDA-graph capture baseline.

Answers "did you try cudnn/ATen with cuda graphs?" -- separates launch-overhead
effects from bandwidth effects, complementing the existing reduce-overhead rows.

For each of the three models (ConvMixer-768/32, ConvMixer-1536/20, RepLKNet-31B),
NCHW and NHWC (channels_last) where supported, fp16, B=1:
  1. eager (uncaptured) baseline, CUDA events, 500 warmup / 500 timed
  2. CUDA-graph-captured replay, 500 warmup (3 on a side stream pre-capture +
     rest via graph replay), 500 timed replays, CUDA events
"""
import argparse
import csv
import os
import sys

import torch

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "convmixer"))
sys.path.insert(0, os.path.join(_REPO, "replknet"))
sys.path.insert(0, os.path.join(_REPO, "replknet", "RepLKNet-pytorch"))

WARMUP = 500
TIMED = 500


def time_eager(model, x, device):
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED):
            model(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED * 1000  # us


def time_cudagraph(model, x, device):
    x_static = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(3):
                model(x_static)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            y_static = model(x_static)

    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED * 1000  # us


def build_convmixer(model_name, layout, device):
    from convmixer import load_pretrained
    channels_last = layout == "NHWC"
    m = load_pretrained(model_name, checkpoints_dir=os.path.join(_REPO, "convmixer", "checkpoints"),
                        device=device, channels_last=channels_last)
    m = m.half().eval()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    return m, x


def build_replknet(layout, device):
    from replknet import create_RepLKNet31B
    CHECKPOINT = os.path.join(_REPO, "replknet", "RepLKNet-31B_ImageNet-1K_224.pth")
    model = create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000,
                                use_checkpoint=False, small_kernel_merged=False)
    if os.path.exists(CHECKPOINT):
        sd = torch.load(CHECKPOINT, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
    model.structural_reparam()
    model.deep_fuse_BN()
    model.eval()
    model = model.to(device).half()
    if layout == "NHWC":
        model = model.to(memory_format=torch.channels_last)
        x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    else:
        model = model.to(memory_format=torch.contiguous_format)
        x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).contiguous()
    return model, x


MODELS = [
    ("ConvMixer-768/32", lambda layout, device: build_convmixer("768_32", layout, device)),
    ("ConvMixer-1536/20", lambda layout, device: build_convmixer("1536_20", layout, device)),
    ("RepLKNet-31B", lambda layout, device: build_replknet(layout, device)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = args.device
    torch.backends.cudnn.benchmark = True

    rows = []
    for name, builder in MODELS:
        for layout in ["NCHW", "NHWC"]:
            print(f"=== {name} {layout} ===", flush=True)
            try:
                model, x = builder(layout, device)
            except Exception as e:
                print(f"  BUILD FAILED: {e}")
                rows.append(dict(model=name, layout=layout, eager_us="", cudagraph_us="",
                                  speedup="", error=f"build: {e}"))
                continue
            try:
                eager_us = time_eager(model, x, device)
                print(f"  eager: {eager_us:.2f} us")
            except Exception as e:
                print(f"  EAGER FAILED: {e}")
                eager_us = None
            try:
                cg_us = time_cudagraph(model, x, device)
                print(f"  cudagraph: {cg_us:.2f} us")
            except Exception as e:
                print(f"  CUDAGRAPH FAILED: {e}")
                cg_us = None
            speedup = (eager_us / cg_us) if (eager_us and cg_us) else ""
            rows.append(dict(
                model=name, layout=layout,
                eager_us=f"{eager_us:.3f}" if eager_us else "",
                cudagraph_us=f"{cg_us:.3f}" if cg_us else "",
                speedup=f"{speedup:.3f}" if speedup != "" else "",
                error="",
            ))
            del model, x
            torch.cuda.empty_cache()

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "layout", "eager_us", "cudagraph_us", "speedup", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
