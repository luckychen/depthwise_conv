#!/usr/bin/env python3
"""
Generate FP16 roofline plot from NCU profiling data.

Reads roofline CSV (results/{ada,spark}_roofline_data.csv) and produces a
log-log roofline plot. Use --platform ada or --platform gb10 to set the
theoretical ceilings correctly.

Usage:
    python plot_roofline.py --input results/ada_roofline_data.csv \\
                            --output results/ada_roofline_plot.png \\
                            --platform ada
    python plot_roofline.py --input results/spark_roofline_data.csv \\
                            --output results/spark_roofline_plot.png \\
                            --platform gb10
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Platform theoretical peaks
PLATFORM_PEAKS = {
    "ada":  {"bw_gbs": 960,  "tflops_fp16": 91.1,  "name": "RTX 6000 Ada"},
    "gb10": {"bw_gbs": 273,  "tflops_fp16": 209,   "name": "NVIDIA GB10"},
}

# Method colors and markers — keys MUST match the method_label values in the CSV
METHOD_STYLES = {
    "Custom NHWC":  {"color": "#1b9e77", "marker": "o", "label": "Custom NHWC", "markersize": 10, "zorder": 5},
    "cuDNN NCHW":   {"color": "#d95f02", "marker": "^", "label": "cuDNN NCHW",  "markersize": 10, "zorder": 4},
    "cuDNN NHWC":   {"color": "#e6ab02", "marker": "s", "label": "cuDNN NHWC",  "markersize": 10, "zorder": 3},
    "Triton NHWC":  {"color": "#7570b3", "marker": "D", "label": "Triton NHWC", "markersize": 10, "zorder": 2},
}


def plot_roofline(data_path, output_path, platform_key):
    peaks = PLATFORM_PEAKS[platform_key]
    peak_bw = peaks["bw_gbs"]
    peak_tflops = peaks["tflops_fp16"]
    plat_name = peaks["name"]

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

    # Roofline ceilings: GFLOP/s = BW_GB_s * AI, capped at peak compute
    ai_range = np.logspace(-1, 3, 200)
    mem_ceiling = peak_bw * ai_range
    mem_ceiling = np.minimum(mem_ceiling, peak_tflops * 1000)
    compute_ceiling = np.full_like(ai_range, peak_tflops * 1000)
    ridge_ai = (peak_tflops * 1000) / peak_bw

    ax.fill_between(ai_range, mem_ceiling, alpha=0.15, color="blue",
                    label=f"Memory bound (BW={peak_bw} GB/s)")
    ax.plot(ai_range, mem_ceiling, color="blue", linewidth=1.5, linestyle="-")

    ax.plot(ai_range, compute_ceiling, color="red", linewidth=1.5, linestyle="--",
            label=f"Compute bound ({peak_tflops} TFLOP/s FP16)")

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
    ax.set_title(f"FP16 Roofline: Depthwise Convolution on {plat_name}\n"
                 "Batch=1, Cold L2",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    # Axis limits
    all_ai = [p["ai"] for p in points]
    all_gf = [p["gflops"] for p in points]
    ax.set_xlim(min(all_ai) * 0.5, max(all_ai) * 2)
    ax.set_ylim(max(min(all_gf) * 0.3, 100), min(max(all_gf) * 1.5, peak_tflops * 1000 * 1.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Roofline plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate FP16 roofline plot")
    parser.add_argument("--input", default="results/spark_roofline_data.csv",
                        help="Input CSV with roofline data")
    parser.add_argument("--output", default="results/spark_roofline_plot.png",
                        help="Output figure path")
    parser.add_argument("--platform", default="gb10", choices=["ada", "gb10"],
                        help="Platform for theoretical peaks and title")
    args = parser.parse_args()

    plot_roofline(args.input, args.output, args.platform)


if __name__ == "__main__":
    main()