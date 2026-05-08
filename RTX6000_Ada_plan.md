# RTX 6000 Ada Experiment Plan — Cross-Platform Pair for Revision

This file is the executable instruction sheet for the RTX 6000 Ada workstation. It pairs with the DGX Spark experiments (`SPARK` branch, `SPARK_context.md`) so the revised paper presents a two-platform L2 residency regime study.

**Read this entire file before doing anything.** Follow the numbered action items exactly.

## 0. Why Ada Matters for the Revision

Reviewer R3 said "single GPU" was a reject-grade flaw. Reviewer R4 said "no conceptual insight." The revision answers both with an **L2 residency principle**: shared-memory weight tiling matters when (working set) > (L2 capacity). Ada (SM 8.9, 142 SMs, **96 MB L2**, 960 GB/s, 91.1 FP16 TFLOPS) and Spark (SM 12.1, 48 SMs, **~24 MB unified**, ~400 GB/s) are deliberately chosen for their 4.4× L2 capacity difference. The expected finding: on Ada, the large L2 absorbs the depthwise working set for most configs, so custom shmem tiling provides diminishing returns relative to Spark. This is the paper's central causal claim.

**Status (2026-05-08)**:
- All experiments: **COMPLETE**. No new Ada experiments needed.
- Paper analysis: L2 hit rate extraction from existing NCU data (Step 5) remains.
- Roofline plot: blocked on matplotlib (Step 5.1).

## 1. Hardware Specs

| Property | Ada (this machine) | Spark |
|---|---|---|
| GPU | NVIDIA RTX 6000 Ada | DGX Spark (GB10) |
| Architecture | Ada Lovelace (SM 8.9) | Blackwell (SM 12.1) |
| SM Count | 142 | 48 |
| L2 Cache | **96 MB** | **~24 MB** (4.4× smaller) |
| Memory | 48 GB GDDR6 | 128 GB unified |
| Memory Bandwidth | 960 GB/s | ~400 GB/s (2.4× lower) |
| FP16 Peak | 91.1 TFLOPS | ~209 TFLOPS |

## 2. Environment

```bash
source /home/chenchon/miniconda3/etc/profile.d/conda.sh && conda activate cuda
export CUDA_VISIBLE_DEVICES=1
cd /home/chenchon/chConv/paper/latex/repo
```

Stack: Python 3.10, PyTorch 2.9.1+cu128, CUDA 12.8 (nvcc 12.9), cuDNN 9.10.2.

### FP16 Methodology

**Use `model.half()` as the primary methodology.** All revision numbers use this. Autocast kept as secondary column for paper continuity.

| Method | How | When |
|--------|-----|------|
| **model.half()** (primary) | `model = model.half()` then `model(x.half())` | All revision numbers |
| autocast (secondary) | FP32 model, `torch.amp.autocast` per forward | Paper continuity only |

## 3. Experiment Results (All Complete)

### 3.1 Per-Kernel Depthwise — ✅

| Config | Custom (μs) | cuDNN NCHW (μs) | cuDNN NHWC (μs) | Triton (μs) | Custom vs best cuDNN |
|--------|------------|-----------------|-----------------|-------------|---------------------|
| 9×9, 1536ch, 32×32 | 28.1 | 90 | 293 | 127 | **3.2×** (vs NCHW) |
| 7×7, 768ch, 32×32 | 27.7 | 27 | **18** | 43 | 0.65× (cuDNN NHWC wins) |
| 31×31, 128ch, 56×56 | 116 | **150** | 634 | 286 | 1.3× |
| 29×29, 256ch, 28×28 | 72 | **77** | 267 | 128 | 1.07× |
| 27×27, 512ch, 14×14 | 22 | **15** | 93 | 38 | 0.68× (cuDNN NCHW wins) |
| 13×13, 1024ch, 7×7 | (native) | **7** | 24 | 22 | 1.0× (native) |

**L2 interpretation**: cuDNN NHWC only wins for 7×7 (specialized kernel exists). For 27×27/512, cuDNN NCHW wins per-kernel (better coalescing on small spatial), but this reverses at network level (see §3.3).

### 3.2 End-to-End Model Latency (model.half(), channels_last, B=1) — ✅

**ConvMixer-1536/20 (9×9, 51.6M params):**

| Configuration | Latency | vs CuDNN best |
|---------------|---------|---------------|
| cuDNN NCHW eager | **3.56 ms** | 1.00× (best cuDNN) |
| cuDNN NHWC eager | 8.84 ms | — |
| cuDNN NHWC +compile(default) | 8.51 ms | — |
| Custom NHWC eager | 2.25 ms | 1.58× |
| Custom NHWC +compile(default) | 1.90 ms | 1.87× |
| **Custom NHWC +compile(r-o)** | **1.89 ms** | **1.88×** |

**ConvMixer-768/32 (7×7, 21.1M params):**

| Configuration | Latency | vs CuDNN best |
|---------------|---------|---------------|
| cuDNN NHWC +compile(r-o) | **1.14 ms** | 1.00× (best cuDNN) |
| cuDNN NHWC eager | 2.52 ms | — |
| cuDNN NCHW eager | 2.25 ms | — |
| Custom NHWC eager | 2.68 ms | 0.43× |
| Custom NHWC +compile(default) | 1.26 ms | 0.90× |
| **Custom NHWC +compile(r-o)** | **0.87 ms** | **1.31×** |

**Key**: 7×7 is the L2 regime boundary. cuDNN has a specialized NHWC kernel here. Custom eager loses (dispatch overhead across 32 layers). But compile eliminates dispatch + CUDA graphs eliminate launch overhead → Custom pulls ahead. On Spark (small L2), Custom wins even at the per-kernel level (25 vs 71 μs).

**RepLKNet-31B (27-31×31, 79M params):**

| Configuration | Latency | vs CuDNN best |
|---------------|---------|---------------|
| cuDNN NCHW eager | **5.09 ms** | 1.00× (best cuDNN) |
| cuDNN NCHW +compile(default) | 6.85 ms | 0.74× (compile hurts!) |
| cuDNN NHWC eager | 7.63 ms | — |
| Custom NHWC eager | 5.18 ms | 0.98× |
| Custom NHWC +compile(default) | 2.51 ms | 2.03× |
| **Custom NHWC +compile(r-o)** | **1.62 ms** | **3.14×** |

### 3.3 Baseline Model Comparison (model.half(), torch.compile reduce-overhead) — ✅

| Model | Params | Top-1 | Latency | Acc/ms |
|-------|--------|-------|---------|--------|
| MobileNetV3-Large | 5.5M | 75.15% | 1.29 ms | 58.3 |
| **ConvMixer-768/32 (custom+r-o)** | **21M** | **80.15%** | **0.87 ms** | **92.1** |
| **RepLKNet-31B (custom+r-o)** | **80M** | **83.30%** | **1.62 ms** | **51.4** |
| **ConvMixer-1536/20 (custom+r-o)** | **52M** | **81.36%** | **1.89 ms** | **43.1** |
| EfficientNet-B0 | 5.3M | 77.68% | 1.70 ms | 45.7 |
| ResNet-152 | 60M | 82.19% | 2.57 ms | 32.0 |

### 3.4 NCU Profiling — ✅ (15/17 profiles)

Full metrics in `results.md` §9. Key L2-relevant findings:

| Config | Method | GFLOP/s | Mem BW (%) | Occupancy | L1 Hit (%) | L2 Hit (%) |
|--------|--------|---------|-----------|-----------|-----------|-----------|
| 9×9, 1536ch | Custom | 3628 | 70.1 | 55.1 | 21.7 | 81.7 |
| 9×9, 1536ch | cuDNN NCHW | 1066 | 80.1 | 62.2 | 99.2 | 51.6 |
| 7×7, 768ch | Custom | 2734 | 51.6 | 63.1 | 90.8 | 75.9 |
| 7×7, 768ch | cuDNN NCHW | 1133 | 52.7 | 63.2 | 97.9 | 54.9 |
| 31×31, 128ch | Custom | 2742 | 59.7 | 32.4 | 89.4 | 95.9 |
| 31×31, 128ch | cuDNN NCHW | 1889 | 78.7 | 61.9 | 99.8 | 76.5 |
| 27×27, 512ch | Custom | 3820 | 33.2 | 30.2 | 73.1 | 93.0 |
| 27×27, 512ch | cuDNN NCHW | 3794 | 52.9 | 63.0 | 99.0 | 41.6 |

**L2 finding**: Custom kernels achieve consistently higher L2 hit rates (76-96%) than cuDNN NCHW (42-77%), especially for large-K configs. Ada's 96MB L2 is large enough that Custom's working set stays mostly resident. On Spark (~24MB), the L2 hit rate gap should be even larger — explaining why Custom's relative advantage grows on small-L2 platforms.

### 3.5 Compile-Survival — ✅

| Method | Eager (μs) | Compiled (μs) | Result |
|--------|-----------|---------------|--------|
| Custom NHWC | 163.7 | 126.0 | **1.30× faster** |
| cuDNN NCHW | 167.7 | **1543.2** | **9.2× degradation** |

**Platform-independent finding.** Confirmed on both Ada (9.2×) and Spark (6.6×).

### 3.6 Batch Sweep (ConvMixer-1536/20, model.half()) — ✅

| B | Custom eager (img/s) | Custom+compile (img/s) | CuDNN eager (img/s) |
|---|----------------------|------------------------|---------------------|
| 1 | 526 | **682** | 135 |
| 4 | **588** | 604 | 125 |
| 8 | 547 | 574 | — |

**L2 interpretation**: Even at B=8, the working set (8 × ~1.5MB per sample ≈ 12MB) fits in Ada's 96MB L2. Batch provides only 12% throughput gain (amortized weight loads) because there's no L2 capacity pressure. On Spark (~24MB), B=4 already pushes the working set near L2 capacity — weight caching should show a larger relative gain. This is the Spark S4 experiment.

### 3.7 Tile Sensitivity (31×31/128ch, M sweep, Cb=32 fixed) — ✅

| M | Shmem | Blocks | Blocks/SM | Latency |
|---|-------|--------|-----------|---------|
| 4 | 60.1 KB | 784 | 5.5 | **0.249 ms** |
| 8 | 60.1 KB | 196 | 1.4 | 0.297 ms |
| 16 | 60.1 KB | 64 | 0.45 | 0.681 ms |

**Finding**: M=4 fastest on Ada (784 blocks saturates 142 SMs). Expected on Spark (48 SMs): M=8 should be optimal (196 blocks → 4.1/SM is adequate). This SM-count-dependent optimal tile is a supporting regime finding.

## 4. Paper Text Update Status

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | ✅ Updated | CUDA events numbers, system-level integration finding |
| §4 (PyTorch Integration) | ✅ Updated | CUDA stream correctness paragraph |
| §5.1 (Methodology) | ✅ Updated | CUDA events per-iteration, bulk timing caveat |
| §5.3 (E2E tables) | ✅ Updated | CUDA events, reduce-overhead rows, 768/32 finding |
| §5.5 (Baselines) | ✅ Updated | All latencies, 768/32 0.87ms |
| §7.3 (Implications) | ✅ Updated | RepLKNet 1.62ms |
| §7.4 (System-Level) | ✅ NEW | Dispatch overhead vs CUDA graphs |
| §8 (Conclusion) | ✅ Updated | Headline numbers |
| **§2.7 (L2 residency analysis)** | **⬜ TODO** | **New subsection — extract L2 metrics from NCU, compare Ada vs Spark** |
| **§5.2 (Table 1 per-kernel)** | **⬜ TODO** | **Expand to 6 configs × 3 methods, add L2 hit column** |
| **§5.4 (NCU analysis)** | **⬜ TODO** | **Center on L2 hit rates as explanatory variable** |
| **§5.6 (Batch scaling)** | **⬜ TODO** | **Add Spark B=1,2,4 column** |
| **§5.7 (Tile sensitivity)** | **⬜ TODO** | **Add Spark column** |
| **§7.1 (Portability)** | **⬜ TODO** | **Replace with L2 residency principle** |

## 5. Remaining Ada Work — Analysis Only (No New Experiments)

### Step 5.1: Install matplotlib + generate roofline plot (~15 min)

```bash
pip install matplotlib
python plot_roofline.py --ncu_csv results/ada_ncu_metrics.csv \
    --output results/ada_roofline_plot.png \
    --fp16_peak_tflops 91.1 --bandwidth_gbps 960
```

### Step 5.2: Extract L2 hit rates from NCU CSVs (~15 min)

The NCU CSV files in `results/ncu/` already contain L2 metrics. Extract `lts__t_sectors` or `l1tex__t_sectors` for each kernel profile. Add to a table:

| Config | Custom L2 hit (%) | cuDNN NCHW L2 hit (%) | cuDNN NHWC L2 hit (%) |
|--------|-------------------|----------------------|----------------------|
| 9×9, 1536ch | 81.7 | 51.6 | (from NCU) |
| 7×7, 768ch | 75.9 | 54.9 | (from NCU) |
| 31×31, 128ch | 95.9 | 76.5 | (from NCU) |
| 27×27, 512ch | 93.0 | 41.6 | (from NCU) |

This table becomes the quantitative backbone of the revised §7.1.

Run:
```bash
python analyze_ncu.py --ncu_dir results/ncu/ --output results/ada_ncu_l2_metrics.csv --metric l2
```

If `analyze_ncu.py` doesn't support `--metric l2`, manually extract from the CSV headers.

### Step 5.3: Build cross-platform L2 comparison table

Once Spark S6 (L2 NCU extraction) is done, merge Ada + Spark L2 data:

| Config | Ada L2 (MB) | Ada Custom L2 hit | Spark L2 (MB) | Spark Custom L2 hit | Δ hit |
|--------|------------|-------------------|---------------|--------------------|--------|
| 9×9, 1536ch | 96 | 81.7% | ~24 | (from Spark NCU) | ? |
| 31×31, 128ch | 96 | 95.9% | ~24 | (from Spark NCU) | ? |
| 27×27, 512ch | 96 | 93.0% | ~24 | (from Spark NCU) | ? |

This is the paper's central evidence table for the L2 residency principle.

## 6. Output Files

```
results/
├── ada_cudnn_layouts.csv                    ✅
├── ada_triton_results.csv                   ✅
├── ada_ncu_metrics.csv                      ✅
├── ada_ncu_tables.md                        ✅
├── ada_ncu_l2_metrics.csv                   ⬜ (Step 5.2)
├── ada_roofline_plot.png                    ⬜ (Step 5.1)
├── ada_cross_platform_l2.csv                ⬜ (Step 5.3, blocks on Spark S6)
├── ada_nsys_compile_fusion.log              ✅
├── ada_1536_20_model_half.log               ✅
├── ada_1536_20_real_model_half_b4.log       ✅
├── ada_1536_20_real_model_half_b8.log       ✅
├── ada_768_32_model_half_b1.log             ✅
├── ada_replknet_model_half_b1.log           ✅
├── ada_baselines_model_half.log             ✅
├── ada_baselines_autocast.log               ✅
├── ada_sensitivity_31x31.csv                ✅
├── ada_sensitivity_31x31.log                ✅
└── ncu/                                     ✅ (15/17 profiles)
```

## 7. Known Pitfalls

1. **Stale ninja lock**: `rm -rf ~/.cache/torch_extensions/py310_cu128/depthwise_*`
2. **torch.compile reduce-overhead on RepLKNet**: First compile 10-20 min. Cache in `/tmp/torchinductor_chenchon/`.
3. **GPU 0 occupied**: Always `CUDA_VISIBLE_DEVICES=1`.
4. **v2 kernel**: Documented dead-end behind `DEPTHWISE_9X9_VARIANT=v2` env toggle. Do not promote. See §9 of previous plan version for full NCU forensics.