"""E1: direct cuDNN graph-API benchmark with heuristic + exhaustive engine search.

Runs depthwise conv (groups=C) for configs C1-C6 x layout{NCHW,NHWC} x batch{1,8},
fp16, using the raw cudnn.pygraph API (bypassing PyTorch's conv dispatch entirely).

For each cell: build with heur_mode {A,B,FALLBACK} -> this is cuDNN's full applicable-engine
candidate list (engine_count() reports raw registered engines, most inapplicable to this
op; A+B+FALLBACK is cuDNN's own find-style exhaustive candidate set). Time every candidate
plan (build_plan_policy.ALL), report the fastest as "exhaustive", and separately report the
first heuristic-mode-A plan's latency (what PyTorch's default heuristic path would pick).
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
LAYOUTS = ["NCHW", "NHWC"]
BATCHES = [1, 8]
WARMUP = 10
TIMED = 50
SEARCH_WARMUP = 3
SEARCH_TIMED = 10
PRECISE_WARMUP = 500
PRECISE_TIMED = 500


def build_tensors(layout, N, C, H, W, K, device):
    pad = K // 2
    x = torch.randn(N, C, H, W, device=device, dtype=torch.float16)
    w = torch.randn(C, 1, K, K, device=device, dtype=torch.float16)
    if layout == "NHWC":
        x = x.to(memory_format=torch.channels_last)
        w = w.to(memory_format=torch.channels_last)
    return x, w, pad


def time_plan(graph, tensor_map, ws, idx, n_runs=TIMED):
    for _ in range(WARMUP):
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


def bench_cell(layout, N, C, H, W, K, device):
    x, w, pad = build_tensors(layout, N, C, H, W, K, device)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        compute_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
    )
    xt = graph.tensor_like(x)
    wt = graph.tensor_like(w)
    y = graph.conv_fprop(image=xt, weight=wt, padding=[pad, pad], stride=[1, 1], dilation=[1, 1])
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
    tensor_map = {xt: x, wt: w, y: y_out}

    # heuristic pick = plan index 0 (top choice under mode A). Fast search-phase timing
    # to rank all plans, then a precise 500-warmup/500-timed re-measurement (context.md
    # methodology) for the heuristic pick and the exhaustive winner only.
    heuristic_engine, heuristic_us, heuristic_name = None, None, None
    try:
        eng_id, _knobs = graph.get_engine_and_knobs_at_index(0)
        heuristic_engine = eng_id
        heuristic_name = graph.get_plan_name_at_index(0)
        heuristic_us = time_plan(graph, tensor_map, ws, 0, n_runs=PRECISE_TIMED)
    except Exception as e:
        heuristic_name = f"FAILED: {e}"

    best_idx, best_us, best_engine, best_name = None, None, None, None
    per_plan = []
    for i in range(n_plans):
        try:
            us = time_plan(graph, tensor_map, ws, i, n_runs=SEARCH_TIMED)
            eng_id, _knobs = graph.get_engine_and_knobs_at_index(i)
            name = graph.get_plan_name_at_index(i)
            per_plan.append((i, eng_id, name, us))
            if best_us is None or us < best_us:
                best_idx, best_us, best_engine, best_name = i, us, eng_id, name
        except Exception:
            continue

    # precise re-time of the search-phase winner
    if best_idx is not None:
        best_us = time_plan(graph, tensor_map, ws, best_idx, n_runs=PRECISE_TIMED)

    per_plan.sort(key=lambda t: t[3])
    top5 = per_plan[:5]

    return dict(
        engine_count=engine_count,
        n_candidates=n_candidates,
        n_supported=n_supported,
        n_plans=n_plans,
        heuristic_engine=heuristic_engine,
        heuristic_name=heuristic_name,
        heuristic_us=heuristic_us,
        best_idx=best_idx,
        best_engine=best_engine,
        best_name=best_name,
        best_us=best_us,
        top5=top5,
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
        for layout in LAYOUTS:
            for B in BATCHES:
                print(f"=== {cfg_name} K={K} C={C} {H}x{W} layout={layout} B={B} ===", flush=True)
                try:
                    r = bench_cell(layout, B, C, H, W, K, device)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    rows.append(dict(
                        config=cfg_name, layout=layout, batch=B, api=f"cudnn_frontend_{frontend_ver}",
                        heuristic_engine="", heuristic_us="", best_engine="", best_us="",
                        n_plans=0, cudnn_version=cudnn_ver, error=str(e),
                    ))
                    continue
                print(f"  engine_count(raw)={r['engine_count']} candidates(A+B+FALLBACK)={r['n_candidates']} "
                      f"supported={r['n_supported']} built={r['n_plans']}")
                print(f"  heuristic: engine={r['heuristic_engine']} name={r['heuristic_name']} us={r['heuristic_us']}")
                print(f"  best:      engine={r['best_engine']} name={r['best_name']} us={r['best_us']}")
                for i, eng, name, us in r["top5"]:
                    print(f"    top5[{i}] engine={eng} name={name} us={us:.2f}")
                rows.append(dict(
                    config=cfg_name, layout=layout, batch=B, api=f"cudnn_frontend_{frontend_ver}",
                    heuristic_engine=r["heuristic_engine"], heuristic_us=f"{r['heuristic_us']:.3f}" if r["heuristic_us"] else "",
                    best_engine=r["best_engine"], best_us=f"{r['best_us']:.3f}" if r["best_us"] else "",
                    n_plans=r["n_plans"], cudnn_version=cudnn_ver, error="",
                ))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "layout", "batch", "api", "heuristic_engine", "heuristic_us",
            "best_engine", "best_us", "n_plans", "cudnn_version", "error",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
