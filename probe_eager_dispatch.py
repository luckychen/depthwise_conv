"""Identify which CUDA kernel actually serves eager depthwise conv per config/layout.

For each C1-C6, B=1, fp16, cudnn.benchmark=True (paper methodology):
- warm up (lets PyTorch autotune pick its winner)
- profile and print the distinct kernel names + per-iter latency
This decides whether the paper's "cuDNN-NCHW eager" baseline is really cuDNN
or ATen's native depthwise kernel, and same for channels_last.
"""
import torch
from torch.profiler import profile, ProfilerActivity

torch.backends.cudnn.benchmark = True
device = "cuda"
CONFIGS = [
    ("C1", 7, 768, 32),
    ("C2", 9, 1536, 32),
    ("C3", 31, 128, 56),
    ("C4", 29, 256, 28),
    ("C5", 27, 512, 14),
    ("C6", 13, 1024, 7),
]

print(f"torch {torch.__version__}  cudnn {torch.backends.cudnn.version()}")

def probe(layout):
    print(f"\n===== layout {layout} =====")
    for name, K, C, HW in CONFIGS:
        conv = torch.nn.Conv2d(C, C, K, padding=K // 2, groups=C, bias=False).to(device).half()
        x = torch.randn(1, C, HW, HW, device=device, dtype=torch.float16)
        if layout == "NHWC":
            conv = conv.to(memory_format=torch.channels_last)
            x = x.to(memory_format=torch.channels_last)
        with torch.no_grad():
            for _ in range(100):
                conv(x)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(500):
                conv(x)
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) / 500 * 1000
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(20):
                    conv(x)
                torch.cuda.synchronize()
        kernels = {}
        for evt in prof.key_averages():
            if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
                kernels[evt.key] = evt.self_device_time_total / 20.0
        kstr = " | ".join(f"{k[:70]} ({v:.1f}us)" for k, v in sorted(kernels.items(), key=lambda t: -t[1])[:3])
        print(f"{name} K={K:2d} C={C:4d} {HW}x{HW}: {us:8.2f} us/iter  kernels: {kstr}")

probe("NCHW")
probe("NHWC")

# Dispatch-identity cross-check that needs no profiler (CUPTI is blocked on GB10):
# if latency is invariant to torch.backends.cudnn.enabled, the op never used cuDNN.
print("\n===== cuDNN on/off invariance (NCHW) =====")
for name, K, C, HW in CONFIGS:
    row = []
    for enabled in (True, False):
        torch.backends.cudnn.enabled = enabled
        torch.backends.cudnn.benchmark = True
        conv = torch.nn.Conv2d(C, C, K, padding=K // 2, groups=C, bias=False).to(device).half()
        x = torch.randn(1, C, HW, HW, device=device, dtype=torch.float16)
        with torch.no_grad():
            for _ in range(100):
                conv(x)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(500):
                conv(x)
            end.record()
            torch.cuda.synchronize()
        row.append(start.elapsed_time(end) / 500 * 1000)
    torch.backends.cudnn.enabled = True
    print(f"{name}: cudnn=on {row[0]:8.2f} us   cudnn=off {row[1]:8.2f} us   ratio {row[0]/row[1]:.3f}")
