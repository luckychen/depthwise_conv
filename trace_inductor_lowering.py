"""E3: Inductor lowering-mechanism trace (re_submit.md).

Establishes, with kernel-name and generated-code evidence, what torch.compile
(mode=default) actually does to the depthwise `aten.convolution` in
ConvMixer-1536/20-NCHW and RepLKNet-31B-NCHW, replacing the factually wrong
"cuDNN kernel decomposed into hundreds of launches" wording.

Run with TORCH_COMPILE_DEBUG=1 so Inductor writes output_code.py under
./torch_compile_debug/; the script greps those files for the convolution
lowering decision (extern_kernels.convolution vs triton templates).
"""
import glob
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "convmixer"))
sys.path.insert(0, os.path.join(_REPO, "replknet"))

device = "cuda"
PROFILE_ITERS = 10


def kernel_table(model, x, tag):
    with torch.no_grad():
        for _ in range(30):
            model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(PROFILE_ITERS):
                model(x)
            torch.cuda.synchronize()
    events = [e for e in prof.key_averages()
              if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0]
    total_launches = sum(e.count for e in events) / PROFILE_ITERS
    print(f"\n--- {tag}: {total_launches:.0f} kernel launches/iter, {len(events)} distinct kernels ---")
    conv_related = []
    for e in sorted(events, key=lambda e: -e.self_device_time_total)[:15]:
        per_iter_us = e.self_device_time_total / PROFILE_ITERS
        count = e.count / PROFILE_ITERS
        print(f"  {e.key[:100]:100s} n/iter={count:6.1f} us/iter={per_iter_us:9.1f}")
    for e in events:
        k = e.key.lower()
        if "conv" in k or "xmma" in k or "cudnn" in k or "implicit_gemm" in k:
            conv_related.append((e.key, e.count / PROFILE_ITERS, e.self_device_time_total / PROFILE_ITERS))
    print(f"  conv-related kernels ({tag}):")
    for k, n, us in conv_related:
        print(f"    {k[:110]} n/iter={n:.1f} us/iter={us:.1f}")
    return total_launches, conv_related


def explain(model, x, tag):
    exp = torch._dynamo.explain(model)(x)
    print(f"\n--- dynamo.explain {tag} ---")
    print(f"  graphs: {exp.graph_count}, graph breaks: {exp.graph_break_count}, "
          f"ops: {sum(len(g.graph.nodes) for g in exp.graphs) if exp.graphs else '?'}")
    for reason in exp.break_reasons[:5]:
        print(f"  break: {str(reason)[:150]}")


def grep_output_code(tag):
    files = sorted(glob.glob("torch_compile_debug/**/output_code.py", recursive=True), key=os.path.getmtime)
    if not files:
        print(f"  [no output_code.py found for {tag}]")
        return
    f = files[-1]
    print(f"\n--- generated code evidence ({tag}): {f} ---")
    extern, triton_kernels = [], 0
    with open(f) as fh:
        for i, line in enumerate(fh, 1):
            if "extern_kernels.convolution" in line or "aten.convolution" in line:
                extern.append((i, line.strip()[:160]))
            if line.startswith("def triton_") or "@triton.jit" in line:
                triton_kernels += 1
    print(f"  extern/aten convolution call sites: {len(extern)}; triton kernel defs: {triton_kernels}")
    for i, line in extern[:8]:
        print(f"    L{i}: {line}")


def run_model(name, build):
    print(f"\n{'='*100}\n===== {name} =====\n{'='*100}")
    model, x = build()
    explain(model, x, name)
    eager_launches, eager_conv = kernel_table(model, x, f"{name} EAGER")
    compiled = torch.compile(model, mode="default")
    with torch.no_grad():
        compiled(x)  # trigger compile before profiling
    comp_launches, comp_conv = kernel_table(compiled, x, f"{name} COMPILED(default)")
    grep_output_code(name)
    print(f"\n>>> {name} SUMMARY: eager {eager_launches:.0f} launches/iter -> compiled {comp_launches:.0f} launches/iter")
    torch._dynamo.reset()
    torch.cuda.empty_cache()


def build_convmixer():
    from convmixer import load_pretrained
    m = load_pretrained("1536_20", checkpoints_dir=os.path.join(_REPO, "convmixer", "checkpoints"),
                        device=device, channels_last=False)
    m = m.to(dtype=torch.float16).eval()
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    return m, x


def build_replknet():
    from benchmark_replknet_nhwc_fp16 import load_model
    m = load_model(device)
    m = m.to(dtype=torch.float16).eval()
    x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)
    return m, x


print(f"torch {torch.__version__} cudnn {torch.backends.cudnn.version()}")
run_model("ConvMixer-1536/20 NCHW fp16", build_convmixer)
run_model("RepLKNet-31B NCHW fp16", build_replknet)
