"""E5: cuDNN graph-API runtime-fusion baseline (conv + bias + activation).

Answers "have you tried cudnn depthwise conv fused with batchnorm, activation etc?"
Builds fused graphs conv(groups=C) -> bias -> GELU, and conv -> scale/shift (folded BN),
for C1-C6, NHWC fp16, B=1, using the same cudnn.pygraph plumbing proven in E1
(benchmark_cudnn_frontend.py). Either fusion is supported (report latency vs the
unfused conv+separate-ops baseline) or it is not (capture the exact error status).
"""
import argparse
import csv
import time

import cudnn
import torch

CONFIGS = [
    ("C1", 7, 768, 32, 32),
    ("C2", 9, 1536, 32, 32),
    ("C3", 31, 128, 56, 56),
    ("C4", 29, 256, 28, 28),
    ("C5", 27, 512, 14, 14),
    ("C6", 13, 1024, 7, 7),
]
WARMUP = 20
TIMED = 200


def time_graph(graph, tensor_map, ws, n_runs=TIMED):
    for _ in range(WARMUP):
        graph.execute(tensor_map, ws)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_runs):
        graph.execute(tensor_map, ws)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_runs * 1000.0  # us


def build_and_time(build_fn, device):
    """build_fn(graph, xt, wt) -> (output_tensor, extra_tensors_dict). Returns (us, error_str)."""
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        compute_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
    )
    try:
        result = build_fn(graph, device)
    except Exception as e:
        return None, f"BUILD: {e}"
    y, tensor_map = result
    try:
        y.set_output(True).set_data_type(cudnn.data_type.HALF)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    except Exception as e:
        return None, f"SUPPORT: {e}"
    ws = torch.empty(graph.get_workspace_size(), device=device, dtype=torch.uint8)
    # allocate an output buffer matching the input's shape/dtype/layout (same-size
    # elementwise-preserving ops: conv with 'same' padding, bias, activation, scale/shift)
    ref = next(iter(tensor_map.values()))
    y_out = torch.empty_like(ref)
    tensor_map[y] = y_out
    try:
        us = time_graph(graph, tensor_map, ws)
        return us, None
    except Exception as e:
        return None, f"EXEC: {e}"


def make_tensors(C, K, H, W, device):
    pad = K // 2
    x = torch.randn(1, C, H, W, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    w = torch.randn(C, 1, K, K, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    b = torch.randn(1, C, 1, 1, device=device, dtype=torch.float16)
    scale = torch.randn(1, C, 1, 1, device=device, dtype=torch.float16)
    shift = torch.randn(1, C, 1, 1, device=device, dtype=torch.float16)
    return x, w, b, scale, shift, pad


def bench_config(cfg_name, K, C, H, W, device):
    x, w, b, scale, shift, pad = make_tensors(C, K, H, W, device)
    out = {}

    # 1. unfused conv only (reference)
    def build_unfused(graph, device):
        xt = graph.tensor_like(x)
        wt = graph.tensor_like(w)
        y = graph.conv_fprop(image=xt, weight=wt, padding=[pad, pad], stride=[1, 1], dilation=[1, 1])
        return y, {xt: x, wt: w}
    us, err = build_and_time(build_unfused, device)
    out["unfused_conv_us"] = us
    out["unfused_conv_error"] = err or ""

    # 2. fused conv -> bias -> gelu
    def build_fused_bias_gelu(graph, device):
        xt = graph.tensor_like(x)
        wt = graph.tensor_like(w)
        bt = graph.tensor_like(b)
        y = graph.conv_fprop(image=xt, weight=wt, padding=[pad, pad], stride=[1, 1], dilation=[1, 1])
        y = graph.bias(input=y, bias=bt)
        y = graph.gelu(input=y)
        return y, {xt: x, wt: w, bt: b}
    us, err = build_and_time(build_fused_bias_gelu, device)
    out["fused_bias_gelu_us"] = us
    out["fused_bias_gelu_error"] = err or ""

    # 3. fused conv -> scale -> shift (folded BN)
    def build_fused_scale_shift(graph, device):
        xt = graph.tensor_like(x)
        wt = graph.tensor_like(w)
        st = graph.tensor_like(scale)
        sht = graph.tensor_like(shift)
        y = graph.conv_fprop(image=xt, weight=wt, padding=[pad, pad], stride=[1, 1], dilation=[1, 1])
        y = graph.mul(a=y, b=st)
        y = graph.add(a=y, b=sht)
        return y, {xt: x, wt: w, st: scale, sht: shift}
    us, err = build_and_time(build_fused_scale_shift, device)
    out["fused_scale_shift_us"] = us
    out["fused_scale_shift_error"] = err or ""

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    device = args.device
    cudnn_ver = cudnn.backend_version_string()
    frontend_ver = cudnn.__version__

    rows = []
    for cfg_name, K, C, H, W in CONFIGS:
        print(f"=== {cfg_name} K={K} C={C} {H}x{W} NHWC ===", flush=True)
        r = bench_config(cfg_name, K, C, H, W, device)
        print(f"  unfused conv:        {r['unfused_conv_us']} us  err={r['unfused_conv_error']}")
        print(f"  fused conv+bias+gelu: {r['fused_bias_gelu_us']} us  err={r['fused_bias_gelu_error']}")
        print(f"  fused conv+scale+shift: {r['fused_scale_shift_us']} us  err={r['fused_scale_shift_error']}")
        rows.append(dict(
            config=cfg_name, K=K, C=C, H=H, W=W, layout="NHWC",
            unfused_conv_us=f"{r['unfused_conv_us']:.3f}" if r["unfused_conv_us"] else "",
            unfused_conv_error=r["unfused_conv_error"],
            fused_bias_gelu_us=f"{r['fused_bias_gelu_us']:.3f}" if r["fused_bias_gelu_us"] else "",
            fused_bias_gelu_error=r["fused_bias_gelu_error"],
            fused_scale_shift_us=f"{r['fused_scale_shift_us']:.3f}" if r["fused_scale_shift_us"] else "",
            fused_scale_shift_error=r["fused_scale_shift_error"],
            cudnn_version=cudnn_ver, frontend_version=frontend_ver,
        ))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "K", "C", "H", "W", "layout",
            "unfused_conv_us", "unfused_conv_error",
            "fused_bias_gelu_us", "fused_bias_gelu_error",
            "fused_scale_shift_us", "fused_scale_shift_error",
            "cudnn_version", "frontend_version",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
