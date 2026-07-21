import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "replknet", "kernels"))
from depthwise_replknet_nhwc_ops import NHWCDepthwise31x31_128, NHWCDepthwise27x27_512

device = "cuda"
WARMUP = 500
TIMED = 500

def bench(module, x):
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

torch.manual_seed(0)

# C3: 31x31, 128ch, 56x56, B=1
m1 = NHWCDepthwise31x31_128()
x1 = torch.randn(1, 128, 56, 56)
us1 = bench(m1, x1)
print(f"C3 (K=31,C=128,56x56,B=1) Custom NHWC: {us1:.3f} us")

# C5: 27x27, 512ch, 14x14, B=1
m2 = NHWCDepthwise27x27_512()
x2 = torch.randn(1, 512, 14, 14)
us2 = bench(m2, x2)
print(f"C5 (K=27,C=512,14x14,B=1) Custom NHWC: {us2:.3f} us")
