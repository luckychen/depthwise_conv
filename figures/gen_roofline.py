"""E7 (re_submit.md Q3, redesigned by analyst 2026-07-22): reuse-band roofline.

Two log-log panels (Ada / GB10). For each config C1-C6, draws a horizontal
arithmetic-intensity BAND from the no-reuse bound (AI=1 FLOP/byte -- every
input element re-fetched per tap, the regime the generic fallback paths
approach) to the ideal-reuse bound (AI = K^2/(2+K^2/HW)), then plots measured
achieved-TFLOPS points for Custom and the ATen-NCHW baseline on that band.

Rationale for the band (not a single point): with ideal reuse, C3/C4/C5 sit
compute-side of Ada's ridge (95 FLOPs/byte) -- R2's "memory-bound regardless
of kernel size" complaint is factually right on Ada under ideal reuse. A
single-point plot would either misrepresent this or invite a "your kernel
is at 8% of peak" attack. The band shows reuse quality determines which wall
is hit, and GB10's much higher ridge (766 FLOPs/byte) makes it bandwidth-side
at every K even under ideal reuse -- a structural cross-platform asymmetry.
"""
import matplotlib.pyplot as plt
import numpy as np

# ---- Peaks ----
ADA_PEAK_TF, ADA_BW_GBS = 91.1, 960.0
GB10_PEAK_TF, GB10_BW_GBS = 209.0, 273.0
ADA_RIDGE = ADA_PEAK_TF * 1e12 / (ADA_BW_GBS * 1e9)
GB10_RIDGE = GB10_PEAK_TF * 1e12 / (GB10_BW_GBS * 1e9)

# ---- Configs: name, K, C, H, W ----
CONFIGS = [
    ("C1", 7, 768, 32, 32),
    ("C2", 9, 1536, 32, 32),
    ("C3", 31, 128, 56, 56),
    ("C4", 29, 256, 28, 28),
    ("C5", 27, 512, 14, 14),
    ("C6", 13, 1024, 7, 7),
]

# ---- Measured warm latencies (us), from results.md sec 2.6/2.9a (Custom) and
#      the eager-dispatch identity probe (ATen-NCHW), both platforms.
#      Ada has NO Custom C6 measurement -- omitted, noted in caption. ----
CUSTOM_ADA = {"C1": 14.494, "C2": 32.282, "C3": 118.7, "C4": 70.291, "C5": 10.9}
CUSTOM_GB10 = {"C1": 29.045, "C2": 82.212, "C3": 294.6, "C4": 137.1, "C5": 44.4, "C6": 190}
ATEN_ADA = {"C1": 23.25, "C2": 91.62, "C3": 152.8, "C4": 75.15, "C5": 14.03, "C6": 8.48}
ATEN_GB10 = {"C1": 70.4, "C2": 292.5, "C3": 411, "C4": 196.8, "C5": 41.9, "C6": 12.45}


def achieved_tflops(K, C, H, W, us):
    if us is None:
        return None
    flops = 2 * K * K * C * H * W
    return flops / (us * 1e-6) / 1e12


def ideal_reuse_ai(K, H, W):
    return (K * K) / (2 + (K * K) / (H * W))


def draw_panel(ax, peak_tf, bw_gbs, ridge, custom_dict, aten_dict, title, flagged):
    ai = np.logspace(-1, 4, 500)
    roof = np.minimum(peak_tf, ai * bw_gbs / 1000.0)  # TFLOPS = min(peak, AI * BW)
    ax.plot(ai, roof, color="black", lw=1.5, zorder=1)
    ax.axhline(peak_tf, color="gray", lw=0.8, ls=":", zorder=0)
    ax.axvline(ridge, color="gray", lw=0.8, ls=":", zorder=0)
    ax.text(ridge * 1.1, peak_tf * 0.06, f"ridge = {ridge:.0f} FLOP/B", fontsize=7, color="gray")

    colors = plt.cm.tab10(np.linspace(0, 1, len(CONFIGS)))
    for (name, K, C, H, W), color in zip(CONFIGS, colors):
        ai_ideal = ideal_reuse_ai(K, H, W)
        ai_noreuse = 1.0
        y_band = min(peak_tf, ai_noreuse * bw_gbs / 1000.0) * 1.0  # band drawn at a nominal low-height row per config
        # horizontal AI band from no-reuse to ideal-reuse, at a fixed measured-achieved height per point instead:
        custom_us = custom_dict.get(name)
        aten_us = aten_dict.get(name)
        custom_tf = achieved_tflops(K, C, H, W, custom_us)
        aten_tf = achieved_tflops(K, C, H, W, aten_us)

        band_y = custom_tf if custom_tf else (aten_tf if aten_tf else 1.0)
        ax.plot([ai_noreuse, ai_ideal], [band_y, band_y], color=color, lw=3, alpha=0.25, zorder=2,
                solid_capstyle="butt")

        if custom_tf:
            ax.scatter([ai_ideal], [custom_tf], color=color, marker="o", s=45, zorder=4,
                       label=f"{name} Custom", edgecolors="black", linewidths=0.4)
        else:
            ax.text(ai_ideal, 1.0, "×", color=color, fontsize=10, ha="center", zorder=4)
        if aten_tf:
            ax.scatter([ai_ideal], [aten_tf], color=color, marker="^", s=40, zorder=3,
                       facecolors="none", edgecolors=color, linewidths=1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title(title)
    ax.set_xlim(1, 1e4)
    ax.set_ylim(0.3, peak_tf * 1.5)
    if flagged:
        for name, note in flagged:
            ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=6.5, color="crimson", va="bottom")


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

draw_panel(axes[0], ADA_PEAK_TF, ADA_BW_GBS, ADA_RIDGE, CUSTOM_ADA, ATEN_ADA,
           "RTX 6000 Ada (91.1 TFLOPS FP16, 960 GB/s)",
           flagged=[("C6", "Note: no Custom C6 measurement on Ada (7² spatial below tiling regime)")])
draw_panel(axes[1], GB10_PEAK_TF, GB10_BW_GBS, GB10_RIDGE, CUSTOM_GB10, ATEN_GB10,
           "DGX Spark / GB10 (209 TFLOPS FP16, 273 GB/s)", flagged=[])

handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", label="Custom (ideal-reuse AI)"),
    plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="none", markeredgecolor="gray", label="ATen-NCHW (ideal-reuse AI)"),
    plt.Line2D([0], [0], color="gray", lw=3, alpha=0.25, label="AI band: no-reuse (1 FLOP/B) → ideal-reuse"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Reuse-band roofline: C1–C6 depthwise conv, B=1 fp16", fontsize=10)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("roofline.pdf", bbox_inches="tight")
print("Wrote roofline.pdf")

# ---- Per-config AI + achieved-TFLOPS table for results.md ----
print("\nconfig,K,C,H,W,AI_noreuse,AI_ideal,custom_ada_tf,aten_ada_tf,custom_gb10_tf,aten_gb10_tf")
for name, K, C, H, W in CONFIGS:
    ai_ideal = ideal_reuse_ai(K, H, W)
    ca = achieved_tflops(K, C, H, W, CUSTOM_ADA.get(name))
    aa = achieved_tflops(K, C, H, W, ATEN_ADA.get(name))
    cg = achieved_tflops(K, C, H, W, CUSTOM_GB10.get(name))
    ag = achieved_tflops(K, C, H, W, ATEN_GB10.get(name))
    def fmt(v):
        return f"{v:.2f}" if v is not None else ""
    print(f"{name},{K},{C},{H},{W},1.0,{ai_ideal:.1f},{fmt(ca)},{fmt(aa)},{fmt(cg)},{fmt(ag)}")
