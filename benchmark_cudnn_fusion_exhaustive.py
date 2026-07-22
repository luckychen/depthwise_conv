"""E5b step 1: exhaustive fused-plan search for conv+bias+GELU (C1, C2, NHWC fp16 B=1).

Reuses the ALL-policy per-plan timing loop proven in E1 (benchmark_cudnn_frontend.py)
to find the true best fused engine, not just the heuristic pick used in E5
(benchmark_cudnn_fusion.py). Makes the fused cuDNN baseline maximally strong
before the paper's Custom numbers are compared against it.
"""
import argparse
import csv

import cudnn
import torch

CONFIGS = [
    ("C1", 7, 768, 32, 32),
    ("C2", 9, 1536, 32, 32),
]
SEARCH_WARMUP = 3
SEARCH_TIMED = 10
PRECISE_WARMUP = 500
PRECISE_TIMED = 500


def time_plan(graph, tensor_map, ws, idx, n_runs):
    for _ in range(SEARCH_WARMUP if n_runs == SEARCH_TIMED else PRECISE_WARMUP):
        graph.execute_plan_at_index(tensor_map, ws, idx)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_runs):
        graph.execute_plan_at_index(tensor_map, ws, idx)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_runs * 1000.0  # us


def bench_config(cfg_name, K, C, H, W, device):
    pad = K // 2
    x = torch.randn(1, C, H, W, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    w = torch.randn(C, 1, K, K, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    b = torch.randn(1, C, 1, 1, device=device, dtype=torch.float16)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        compute_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
    )
    xt = graph.tensor_like(x)
    wt = graph.tensor_like(w)
    bt = graph.tensor_like(b)
    y = graph.conv_fprop(image=xt, weight=wt, padding=[pad, pad], stride=[1, 1], dilation=[1, 1])
    y = graph.bias(input=y, bias=bt)
    y = graph.gelu(input=y)
    y.set_output(True).set_data_type(cudnn.data_type.HALF)
    graph.validate()
    graph.build_operation_graph()

    engine_count = graph.get_engine_count()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.FALLBACK])
    n_candidates = graph.get_execution_plan_count()
    graph.check_support()
    n_supported = graph.get_execution_plan_count()
    graph.build_plans(cudnn.build_plan_policy.ALL)
    n_plans = graph.get_execution_plan_count()

    y_out = torch.empty_like(x)
    ws = torch.empty(graph.get_workspace_size(), device=device, dtype=torch.uint8)
    tensor_map = {xt: x, wt: w, bt: b, y: y_out}

    # heuristic pick = plan index 0, precisely re-timed
    heuristic_engine, heuristic_us = None, None
    try:
        eng_id, _knobs = graph.get_engine_and_knobs_at_index(0)
        heuristic_engine = eng_id
        heuristic_us = time_plan(graph, tensor_map, ws, 0, PRECISE_TIMED)
    except Exception as e:
        print(f"  heuristic plan 0 failed: {e}")

    best_idx, best_us, best_engine = None, None, None
    per_plan = []
    for i in range(n_plans):
        try:
            us = time_plan(graph, tensor_map, ws, i, SEARCH_TIMED)
            eng_id, _knobs = graph.get_engine_and_knobs_at_index(i)
            per_plan.append((i, eng_id, us))
            if best_us is None or us < best_us:
                best_idx, best_us, best_engine = i, us, eng_id
        except Exception:
            continue
    if best_idx is not None:
        best_us = time_plan(graph, tensor_map, ws, best_idx, PRECISE_TIMED)

    per_plan.sort(key=lambda t: t[2])
    print(f"  engine_count(raw)={engine_count} candidates={n_candidates} supported={n_supported} built={n_plans}")
    print(f"  heuristic: engine={heuristic_engine} us={heuristic_us}")
    print(f"  best:      engine={best_engine} us={best_us}")
    for i, eng, us in per_plan[:5]:
        print(f"    top5[{i}] engine={eng} us={us:.2f}")

    return dict(
        heuristic_engine=heuristic_engine, heuristic_us=heuristic_us,
        best_engine=best_engine, best_us=best_us, n_plans=n_plans,
    )


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
        print(f"=== {cfg_name} K={K} C={C} {H}x{W} NHWC fused conv+bias+gelu ===", flush=True)
        try:
            r = bench_config(cfg_name, K, C, H, W, device)
        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append(dict(config=cfg_name, K=K, C=C, H=H, W=W,
                              heuristic_engine="", heuristic_us="", best_engine="", best_us="",
                              n_plans=0, cudnn_version=cudnn_ver, frontend_version=frontend_ver, error=str(e)))
            continue
        rows.append(dict(
            config=cfg_name, K=K, C=C, H=H, W=W,
            heuristic_engine=r["heuristic_engine"],
            heuristic_us=f"{r['heuristic_us']:.3f}" if r["heuristic_us"] else "",
            best_engine=r["best_engine"],
            best_us=f"{r['best_us']:.3f}" if r["best_us"] else "",
            n_plans=r["n_plans"], cudnn_version=cudnn_ver, frontend_version=frontend_ver, error="",
        ))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "K", "C", "H", "W",
            "heuristic_engine", "heuristic_us", "best_engine", "best_us",
            "n_plans", "cudnn_version", "frontend_version", "error",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
