#!/usr/bin/env python3
"""
Generate FP16 roofline plot from NCU profiling data.

Reads spark_roofline_data.csv and produces a log-log roofline plot
with theoretical ceilings for the NVIDIA GB10 (Grace Blackwell).

Usage:
    python plot_roofline.py [--input results/spark_roofline_data.csv]
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# GB10 theoretical peaks (estimates; NCU SpeedOfLight will give actuals)
# From NCU data: SM Frequency ~2.1 GHz, 48 SMs
# FP16 non-tensor-core throughput: we derive from NCU SpeedOfLight data
PEAK_MEM_BW_GBS = 960     # RTX 6000 Ada GDDR6
PEAK_FP16_TFLOPS = 91.1   # RTX 6000 Ada FP16 peak

# Method colors and markers
METHOD_STYLES = {
    "custom":        {"color": "#1b9e77", "marker": "o", "label": "Custom NHWC", "markersize": 10, "zorder": 5},
    "cudnn_nchw":    {"color": "#d95f02", "marker": "^", "label": "cuDNN NCHW", "markersize": 10, "zorder": 4},
    "cudnn_nhwc":    {"color": "#e6ab02", "marker": "s", "label": "cuDNN NHWC", "markersize": 10, "zorder": 3},
    "triton_nhwc":   {"color": "#7570b3", "marker": "D", "label": "Triton NHWC", "markersize": 10, "zorder": 2},
}


def plot_roofline(data_path, output_path):
    # Read data
    points = []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ai = float(row.get("AI_flop_per_byte", 0))
            gflops = float(row.get("GFLOP_s", 0))
            if ai > 0 and gflops > 0:
                points.append({
                    "ai": ai,
                    "gflops": gflops,
                    "method": row["method_label"],
                    "config": row["config_label"],
                    "mem_bw": float(row.get("MemBW", 0)),
                    "sm_thru": float(row.get("SMThru", 0)),
                    "occ": float(row.get("AchievedOcc", 0)),
                })

    if not points:
        print("No valid roofline points found.")
        sys.exit(1)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Roofline ceilings
    # Memory bandwidth ceiling: GFLOP/s = BW_GB_s * AI
    ai_range = np.logspace(-1, 3, 200)
    mem_ceiling = PEAK_MEM_BW_GBS * ai_range
    # Cap at peak compute
    mem_ceiling = np.minimum(mem_ceiling, PEAK_FP16_TFLOPS * 1000)

    compute_ceiling = np.full_like(ai_range, PEAK_FP16_TFLOPS * 1000)

    # Ridge point (where memory and compute ceilings meet)
    ridge_ai = (PEAK_FP16_TFLOPS * 1000) / PEAK_MEM_BW_GBS

    ax.fill_between(ai_range, mem_ceiling, alpha=0.15, color="blue",
                    label=f"Memory bound (BW={PEAK_MEM_BW_GBS} GB/s)")
    ax.plot(ai_range, mem_ceiling, color="blue", linewidth=1.5, linestyle="-")

    # Compute ceiling (horizontal)
    ax.plot(ai_range, compute_ceiling, color="red", linewidth=1.5, linestyle="--",
            label=f"Compute bound ({PEAK_FP16_TFLOPS} TFLOP/s FP16)")

    # Ridge line
    ax.axvline(x=ridge_ai, color="gray", linewidth=0.8, linestyle=":",
               alpha=0.7)

    # Plot data points grouped by method
    for method_key, style in METHOD_STYLES.items():
        method_pts = [p for p in points if p["method"] == method_key]
        if not method_pts:
            continue

        ais = [p["ai"] for p in method_pts]
        gfs = [p["gflops"] for p in method_pts]

        ax.scatter(ais, gfs,
                   c=style["color"], marker=style["marker"],
                   s=style["markersize"]**2,
                   label=style["label"],
                   zorder=style["zorder"],
                   edgecolors="black", linewidth=0.5)

        # Annotate each point with config label (short form)
        for p in method_pts:
            # Short label
            parts = p["config"].split(",")
            short = parts[0].strip().replace("×", "x") if parts else ""
            ax.annotate(short,
                        (p["ai"], p["gflops"]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=7,
                        alpha=0.85,
                        color=style["color"])

    # Styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Achieved Performance (GFLOP/s)", fontsize=12)
    ax.set_title("FP16 Roofline: Depthwise Convolution on NVIDIA GB10\n"
                 "Batch=1, Cold L2",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    # Axis limits
    all_ai = [p["ai"] for p in points]
    all_gf = [p["gflops"] for p in points]
    ax.set_xlim(min(all_ai) * 0.5, max(all_ai) * 2)
    ax.set_ylim(max(min(all_gf) * 0.3, 100), min(max(all_gf) * 1.5, PEAK_FP16_TFLOPS * 1000 * 1.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Roofline plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate FP16 roofline plot")
    parser.add_argument("--input", default="results/spark_roofline_data.csv",
                        help="Input CSV with roofline data")
    parser.add_argument("--output", default="results/spark_roofline_plot.png",
                        help="Output figure path")
    args = parser.parse_args()

    plot_roofline(args.input, args.output)


if __name__ == "__main__":
    main()