#!/usr/bin/env python3
"""
analyze_ncu.py — Parse NCU CSV profiles and generate paper-ready tables.

Reads all NCU CSV files from results/ncu/, extracts key performance metrics,
and produces:
  - results/spark_ncu_metrics.csv  — per-kernel metrics
  - results/spark_roofline_data.csv — arithmetic intensity + GFLOP/s
  - results/spark_ncu_tables.md    — paper-ready markdown tables

Usage:
    python analyze_ncu.py [--ncu-dir results/ncu]
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict


# --- config definitions -------------------------------------------------------

CONFIGS = {
    "9x9_1536_32":  {"C": 1536, "K": 9,  "H": 32, "W": 32, "label": "9×9, 1536ch, 32×32"},
    "7x7_768_32":   {"C": 768,  "K": 7,  "H": 32, "W": 32, "label": "7×7, 768ch, 32×32"},
    "31x31_128_56": {"C": 128,  "K": 31, "H": 56, "W": 56, "label": "31×31, 128ch, 56×56"},
    "29x29_256_28": {"C": 256,  "K": 29, "H": 28, "W": 28, "label": "29×29, 256ch, 28×28"},
    "27x27_512_14": {"C": 512,  "K": 27, "H": 14, "W": 14, "label": "27×27, 512ch, 14×14"},
    "13x13_1024_7": {"C": 1024, "K": 13, "H": 7,  "W": 7,  "label": "13×13, 1024ch, 7×7"},
}

METHOD_LABELS = {
    "custom":        "Custom NHWC",
    "cudnn_nchw":    "cuDNN NCHW",
    "cudnn_nhwc":    "cuDNN NHWC",
    "triton_nhwc":   "Triton NHWC",
}

# key metrics to extract (NCU section, metric name) -> short name
KEY_METRICS = [
    ("GPU Speed Of Light Throughput", "Memory Throughput", "%", "MemBW"),
    ("GPU Speed Of Light Throughput", "Compute (SM) Throughput", "%", "SMThru"),
    ("GPU Speed Of Light Throughput", "Duration", "ns", "Duration_ns"),
    ("GPU Speed Of Light Throughput", "Elapsed Cycles", "cycle", "Cycles"),
    ("GPU Speed Of Light Throughput", "SM Frequency", "hz", "SMFreq_hz"),
    ("GPU Speed Of Light Throughput", "L1/TEX Cache Throughput", "%", "L1Thru"),
    ("GPU Speed Of Light Throughput", "L2 Cache Throughput", "%", "L2Thru"),
    ("Memory Workload Analysis", "Mem Busy", "%", "MemBusy"),
    ("Memory Workload Analysis", "Max Bandwidth", "%", "MaxBW"),
    ("Memory Workload Analysis", "L1/TEX Hit Rate", "%", "L1HitRate"),
    ("Memory Workload Analysis", "L2 Hit Rate", "%", "L2HitRate"),
    ("Memory Workload Analysis", "Mem Pipes Busy", "%", "MemPipesBusy"),
    ("Occupancy", "Achieved Occupancy", "%", "AchievedOcc"),
    ("Occupancy", "Theoretical Occupancy", "%", "TheoOcc"),
    ("Occupancy", "Achieved Active Warps Per SM", "warp", "ActiveWarpsSM"),
    ("Occupancy", "Block Limit Shared Mem", "block", "BlockLimitShmem"),
    ("Occupancy", "Block Limit Registers", "block", "BlockLimitRegs"),
    ("Occupancy", "Block Limit Warps", "block", "BlockLimitWarps"),
    ("Compute Workload Analysis", "SM Busy", "%", "SMBusy"),
    ("Compute Workload Analysis", "Executed Ipc Active", "inst/cycle", "IPC"),
    ("Scheduler Statistics", "No Eligible", "%", "StallNoEligible"),
    ("Scheduler Statistics", "Active Warps Per Scheduler", "warp", "ActiveWarpsSched"),
    ("Warp State Statistics", "Warp Cycles Per Issued Instruction", "cycle", "CyclesPerInst"),
    ("Launch Statistics", "Grid Size", "", "GridSize"),
    ("Launch Statistics", "Block Size", "", "BlockSize"),
    ("Launch Statistics", "Waves Per SM", "", "WavesPerSM"),
    ("Launch Statistics", "Dynamic Shared Memory Per Block", "byte/block", "DynShmemBytes"),
    ("Launch Statistics", "Registers Per Thread", "register/thread", "RegsPerThread"),
    ("Source Counters", "Branch Instructions Ratio", "%", "BranchRatio"),
]


def parse_ncu_csv(filepath):
    """Parse an NCU CSV file and return a dict of {metric_short_name: value}."""
    raw_lines = []
    with open(filepath) as f:
        for line in f:
            if not line.startswith("=="):
                raw_lines.append(line)

    # Find the header
    header_idx = None
    for i, line in enumerate(raw_lines):
        if line.startswith('"ID"'):
            header_idx = i
            break

    if header_idx is None:
        return {}

    reader = csv.DictReader(raw_lines[header_idx:])
    data = {}
    for row in reader:
        section = row.get("Section Name", "").strip()
        metric = row.get("Metric Name", "").strip()
        unit = row.get("Metric Unit", "").strip()
        value_str = row.get("Metric Value", "").strip().replace(",", "")

        # Skip empty
        if not metric or not value_str:
            continue

        # Map to short name
        for sec, met, u, short in KEY_METRICS:
            if section == sec and metric == met and unit == u:
                try:
                    data[short] = float(value_str) if value_str else 0.0
                except ValueError:
                    pass
                break

    return data


def compute_arithmetic_intensity(cfg, mem_pct, mem_peak_gbs=200):
    """Compute arithmetic intensity (FLOP/byte) for roofline.

    Depthwise conv FLOP count: H*W*C*K*K*2 (multiply-add per element)
    We estimate DRAM bytes from bandwidth utilization and duration.

    Returns: (AI_flop_per_byte, GFLOP_s)
    """
    C, K, H, W = cfg["C"], cfg["K"], cfg["H"], cfg["W"]
    flops = H * W * C * K * K * 2  # total FLOP

    # From NCU we get bandwidth utilization — need to estimate bytes
    # This is a rough estimate; NCU SpeedOfLight would give exact values
    theoretical_bytes_in = (H * W * C + K * K * C) * 2  # FP16 input + weights
    theoretical_bytes_out = H * W * C * 2  # FP16 output

    return flops, theoretical_bytes_in + theoretical_bytes_out


def main():
    parser = argparse.ArgumentParser(description="Analyze NCU profiling CSV files")
    parser.add_argument("--ncu-dir", default="results/ncu",
                        help="Directory containing NCU CSV files")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for output files")
    args = parser.parse_args()

    ncu_dir = args.ncu_dir
    output_dir = args.output_dir

    # Parse all NCU CSV files
    csv_files = sorted(glob.glob(os.path.join(ncu_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {ncu_dir}")
        sys.exit(1)

    print(f"Parsing {len(csv_files)} NCU CSV files...")

    results = []
    for fpath in csv_files:
        basename = os.path.basename(fpath).replace(".csv", "")
        # Parse filename: method_config.csv where method can be multi-word
        # Try method prefixes: triton_nhwc, cudnn_nchw, cudnn_nhwc, custom
        method = None
        config_key = None
        for candidate in ["triton_nhwc", "cudnn_nchw", "cudnn_nhwc", "custom"]:
            prefix = candidate + "_"
            if basename.startswith(prefix):
                method = candidate
                config_key = basename[len(prefix):]
                break

        if config_key not in CONFIGS:
            print(f"  Skipping unknown config: {basename}")
            continue

        cfg = CONFIGS[config_key]
        metrics = parse_ncu_csv(fpath)

        if not metrics:
            print(f"  Warning: no metrics extracted from {basename}")
            continue

        row = {
            "file": basename,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "config": config_key,
            "config_label": cfg["label"],
            "C": cfg["C"],
            "K": cfg["K"],
            "H": cfg["H"],
            "W": cfg["W"],
        }
        row.update(metrics)

        # Compute roofline coordinates
        flops, est_bytes = compute_arithmetic_intensity(cfg, metrics.get("MaxBW", 0))
        duration_ns = metrics.get("Duration_ns", 1)
        gflops_per_s = (flops / (duration_ns * 1e-9)) / 1e9 if duration_ns > 0 else 0

        # Estimate AI: if we have memory throughput %, we can estimate actual bytes
        mem_thru_pct = metrics.get("MaxBW", 50)  # % of peak bandwidth
        # Assume peak bandwidth ~400 GB/s for GB10 (NVLink-C2C)
        # Actual bytes moved = bandwidth * time * utilization
        peak_bw_gbs = 400  # GB/s estimate for GB10 unified memory
        actual_bytes = peak_bw_gbs * 1e9 * (duration_ns * 1e-9) * (mem_thru_pct / 100.0)
        if actual_bytes > 0:
            ai = flops / actual_bytes
        else:
            ai = flops / max(est_bytes, 1)

        row["GFLOP_s"] = round(gflops_per_s, 2)
        row["AI_flop_per_byte"] = round(ai, 2)

        results.append(row)
        print(f"  {basename}: dur={metrics.get('Duration_ns','?'):.0f}ns  "
              f"mem={metrics.get('MemBW','?'):.1f}%  "
              f"occ={metrics.get('AchievedOcc','?'):.1f}%  "
              f"AI={ai:.1f}  GFLOP/s={gflops_per_s:.1f}")

    if not results:
        print("No valid results extracted.")
        sys.exit(1)

    # === Output 1: spark_ncu_metrics.csv ===
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "spark_ncu_metrics.csv")

    # Collect all column names
    all_cols = ["config_label", "method_label", "C", "K", "H", "W",
                "GFLOP_s", "AI_flop_per_byte"]
    for _, _, _, short in KEY_METRICS:
        all_cols.append(short)

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nMetrics table written: {metrics_path}  ({len(results)} rows)")

    # === Output 2: spark_roofline_data.csv ===
    roofline_path = os.path.join(output_dir, "spark_roofline_data.csv")
    rl_cols = ["config_label", "method_label", "C", "K", "H", "W",
               "GFLOP_s", "AI_flop_per_byte", "MemBW", "SMThru", "AchievedOcc"]
    with open(roofline_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rl_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Roofline data written: {roofline_path}")

    # === Output 3: spark_ncu_tables.md ===
    tables_path = os.path.join(output_dir, "spark_ncu_tables.md")
    with open(tables_path, "w") as f:
        f.write("# NCU Profiling Results — Microarchitecture Characterization\n\n")
        f.write(f"Platform: NVIDIA GB10 (SM 12.1, 48 SMs)\n")
        f.write(f"Conditions: batch=1, FP16, cold L2\n\n")

        # Table 1: SpeedOfLight + Occupancy (all kernels)
        f.write("## Table 1: SpeedOfLight and Occupancy\n\n")
        f.write("| Config | Method | Mem BW (%) | SM Thr. (%) | Achieved Occ (%) | Theo Occ (%) | Duration (μs) |\n")
        f.write("|--------|--------|-----------|------------|-----------------|-------------|---------------|\n")
        for r in sorted(results, key=lambda x: (x["C"], x["K"])):
            dur_us = r.get("Duration_ns", 0) / 1000.0
            f.write(f"| {r['config_label']} | {r['method_label']} | "
                    f"{r.get('MemBW', 0):.1f} | {r.get('SMThru', 0):.1f} | "
                    f"{r.get('AchievedOcc', 0):.1f} | {r.get('TheoOcc', 0):.1f} | "
                    f"{dur_us:.1f} |\n")
        f.write("\n")

        # Table 2: Memory Hierarchy
        f.write("## Table 2: Memory Hierarchy\n\n")
        f.write("| Config | Method | L1 Hit (%) | L2 Hit (%) | Mem Busy (%) | Max BW (%) | Shmem/Block (KB) |\n")
        f.write("|--------|--------|-----------|-----------|-------------|-----------|------------------|\n")
        for r in sorted(results, key=lambda x: (x["C"], x["K"])):
            shmem_kb = r.get("DynShmemBytes", 0) / 1024.0
            f.write(f"| {r['config_label']} | {r['method_label']} | "
                    f"{r.get('L1HitRate', 0):.1f} | {r.get('L2HitRate', 0):.1f} | "
                    f"{r.get('MemBusy', 0):.1f} | {r.get('MaxBW', 0):.1f} | "
                    f"{shmem_kb:.1f} |\n")
        f.write("\n")

        # Table 3: Stall Analysis
        f.write("## Table 3: Stall and Scheduler Analysis\n\n")
        f.write("| Config | Method | No Eligible (%) | IPC | Cycles/Inst | Waves/SM | Branch (%) |\n")
        f.write("|--------|--------|----------------|-----|-------------|----------|-----------|\n")
        for r in sorted(results, key=lambda x: (x["C"], x["K"])):
            f.write(f"| {r['config_label']} | {r['method_label']} | "
                    f"{r.get('StallNoEligible', 0):.1f} | {r.get('IPC', 0):.2f} | "
                    f"{r.get('CyclesPerInst', 0):.1f} | {r.get('WavesPerSM', 0):.1f} | "
                    f"{r.get('BranchRatio', 0):.1f} |\n")
        f.write("\n")

        # Table 4: Roofline coordinates
        f.write("## Table 4: Roofline Coordinates\n\n")
        f.write("| Config | Method | AI (FLOP/byte) | GFLOP/s | Duration (μs) |\n")
        f.write("|--------|--------|---------------|---------|---------------|\n")
        for r in sorted(results, key=lambda x: (x["C"], x["K"])):
            dur_us = r.get("Duration_ns", 0) / 1000.0
            f.write(f"| {r['config_label']} | {r['method_label']} | "
                    f"{r.get('AI_flop_per_byte', 0):.1f} | {r.get('GFLOP_s', 0):.1f} | "
                    f"{dur_us:.1f} |\n")
        f.write("\n")

        # Analysis notes
        f.write("## Analysis Notes\n\n")
        f.write("### Key Observations\n\n")

        # Find fastest per config
        by_config = defaultdict(list)
        for r in results:
            by_config[r["config_label"]].append(r)
        for label, rows in sorted(by_config.items()):
            best = min(rows, key=lambda x: x.get("Duration_ns", float("inf")))
            f.write(f"- **{label}**: Fastest is {best['method_label']} "
                    f"({best.get('Duration_ns', 0)/1000:.1f} μs)\n")

    print(f"Markdown tables written: {tables_path}")


if __name__ == "__main__":
    main()