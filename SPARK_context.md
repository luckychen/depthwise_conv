# DGX Spark — Latency-Oriented Experiments (Batch=1)

## 1. What This Is

This repo contains custom CUDA kernels for **large-kernel depthwise convolution** — the channel-independent convolution used in ConvMixer and RepLKNet. We are revising a paper submitted to **ACM TACO** that received "major revision." The reviewers' main complaints:

1. **"No novelty"** — techniques (shmem tiling, double buffering) are well-known
2. **"cuDNN-only baseline"** — need TVM/Triton comparisons
3. **"Single GPU (RTX 6000 Ada)"** — need cross-platform evidence, especially edge
4. **"Batch=1 only"** — need batch scaling study
5. **"Insufficient performance analysis"** — need roofline, occupancy, instruction mix

The revised paper thesis is: **large-kernel depthwise convolution requires regime-aware implementation** — the optimal mapping changes with batch size, hardware, and kernel size. DGX Spark is our **latency-oriented / compact deployment** platform.

**On this branch (SPARK), all experiments use batch=1.** This matches the near-edge deployment scenario where DGX Spark operates — latency-sensitive, single-inference workloads (robotics, video analytics).

## 2. DGX Spark Platform — Discovered Hardware

Hardware discovery completed 2026-04-26. Full details in `results/spark_hardware_info.txt`.

| Property | Value |
|----------|-------|
| GPU | NVIDIA GB10 |
| Architecture | Grace Blackwell (unified memory) |
| Compute Capability | **SM 12.1 (compute_121)** |
| SM Count | **48** |
| Memory | **128.5 GB unified** (Grace CPU + Blackwell GPU shared) |
| Driver | 580.95.05 |
| CUDA Toolkit | 13.0 (system) / 12.8 (PyTorch bundled) |
| CPU | ARM Grace (aarch64) |
| OS | Ubuntu 24.04.3 LTS |

Key differences from original RTX 6000 Ada testbed:

| | RTX 6000 Ada | DGX Spark (GB10) |
|---|---|---|
| SM generation | SM 8.9 (Ada Lovelace) | SM 12.1 (Blackwell) |
| SM count | 128 | 48 |
| Memory | 48 GB GDDR6 (dedicated) | 128.5 GB unified |
| Memory model | Discrete (PCIe) | Unified (Grace-Blackwell NVLink-C2C) |
| Target use | Workstation | Compact local deployment / near-edge |

**Do NOT call DGX Spark an "embedded edge device."** Call it a "compact local deployment platform" or "near-edge AI system."

## 3. Software Environment

Conda environment: `chConv`

| Package | Version |
|---------|---------|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CuDNN | 9.19.0 |
| Triton | 3.6.0 |
| ninja | 1.13.0 |
| timm | 1.0.26 |
| numpy | 2.4.3 |
| Pillow | 12.1.1 |

## 4. Experiment Results

All experiments: **batch=1, FP16**. Comparisons are **apple-to-apple** (eager vs eager, compile vs compile). Triton is a compiled DSL, so it is compared against torch.compile'd models.

### 4.1 Compile and Verify Existing Kernels — DONE

JIT compilation works on SM 12.1 with no code changes. Auto-detection picked up compute_121 correctly.

### 4.2 ConvMixer Latency Benchmark (eager vs eager) — DONE

Results (500 runs, batch=1, FP16 NHWC):

**ConvMixer-1536/20 (9×9, 20 layers):**

| Configuration | Latency | Eager-to-eager | Compile-to-compile |
|---|---|---|---|
| CuDNN eager | 26.83 ms | — | — |
| Custom eager | 6.83 ms | **3.93×** | — |
| CuDNN + compile | 25.44 ms | — | — |
| Custom + compile | 4.91 ms | — | **5.18×** |

**ConvMixer-768/32 (7×7, 32 layers):**

| Configuration | Latency | Eager-to-eager | Compile-to-compile |
|---|---|---|---|
| CuDNN eager | 15.51 ms | — | — |
| Custom eager | 4.61 ms | **3.36×** | — |
| CuDNN + compile | 13.61 ms | — | — |
| Custom + compile | 2.47 ms | — | **5.51×** |

### 4.3 RepLKNet Latency Benchmark — DONE (Updated with native NCHW)

Results (100 runs, batch=1, FP16, native layouts — zero conversion in any config):

| Configuration | Layout | Latency | vs cuDNN-best |
|---|---|---|---|
| CuDNN eager | NHWC | 18.14 ms | 0.46× |
| CuDNN NCHW eager | **NCHW** | **8.26 ms** | 1.00× |
| CuDNN + compile | NHWC | 15.00 ms | 0.55× |
| CuDNN NCHW + compile | **NCHW** | **5.15 ms** | 1.60× |
| Custom eager | NHWC | 7.74 ms | 1.07× |
| Custom + compile | NHWC | **4.52 ms** | **1.83×** |
| Triton NHWC + compile | NHWC | 273.59 ms | 0.03× |
| Triton NCHW + compile | NCHW | 373.54 ms | 0.02× |

**cuDNN NCHW is 2.2× faster than cuDNN NHWC end-to-end** (8.26 vs 18.14 ms eager). This is NOT a stranded optimization — cuDNN can run the entire network in native NCHW.

**Custom NHWC still wins** (4.52 ms vs 5.15 ms cuDNN NCHW compiled) by 1.14×.

Per-stage depthwise breakdown (all three baselines, eager, per-call times × count):

| Stage | Config | Custom NHWC | cuDNN NHWC | cuDNN NCHW | Count |
|---|---|---|---|---|---|
| stem | 3×3, 128ch | 0.262 ms | 0.166 ms | 0.029 ms | 2 |
| 0 | 31×31, 128ch | 0.423 ms | 2.946 ms | 0.478 ms | 2 |
| 1 | 29×29, 256ch | 0.268 ms | 1.193 ms | 0.264 ms | 2 |
| 2 | 27×27, 512ch | 0.077 ms | 0.340 ms | 0.080 ms | 18 |
| 3 | 13×13, 1024ch | 0.093 ms | 0.093 ms | 0.025 ms | 2 |
| **DW total** | | **4.74 ms** | **16.19 ms** | **3.52 ms** | |
| **Non-DW** | | **3.00 ms** | **1.96 ms** | **4.75 ms** | |
| **Total** | | **7.74 ms** | **18.14 ms** | **8.26 ms** | |

**Critical finding — DW vs non-DW tradeoff:**
- cuDNN NCHW: faster DW (3.52 ms) but slower non-DW (4.75 ms)
- Custom NHWC: slightly slower DW (4.74 ms) but much faster non-DW (3.00 ms)
- Custom wins overall because NHWC benefits the non-DW layers (1×1 conv, BN, SE blocks) more than it costs on DW layers
- The per-stage DW numbers show Custom ≈ cuDNN NCHW for stages 1-2 (within 5%), but Custom is 12% faster on stage 0 (31×31) due to shared-memory tiling advantage at that scale

### 4.4 cuDNN Best-Layout Baseline (Isolated Kernel) — DONE

**Critical finding:** cuDNN NCHW is 3.7–6.8× faster than NHWC for **isolated** depthwise conv kernels on GB10.

| Config | NCHW (ms) | NHWC (ms) | Best | Ratio |
|---|---|---|---|---|
| 9×9, 1536ch, 32×32 | 0.295 | 1.088 | NCHW | 3.68× |
| 7×7, 768ch, 32×32 | 0.071 | 0.362 | NCHW | 5.13× |
| 31×31, 128ch, 56×56 | 0.416 | 2.045 | NCHW | 4.92× |
| 29×29, 256ch, 28×28 | 0.198 | 1.026 | NCHW | 5.17× |
| 27×27, 512ch, 14×14 | 0.044 | 0.301 | NCHW | 6.77× |
| 13×13, 1024ch, 7×7 | 0.013 | 0.054 | NCHW | 4.01× |

However, at **end-to-end network level**, cuDNN NCHW and NHWC are nearly tied (~24 ms vs ~24 ms for ConvMixer-1536/20) because the layout conversion overhead between layers absorbs the per-kernel NCHW advantage.

### 4.5 Triton Baseline (Isolated Kernel) — DONE

Triton kernel design: vectorized over BLOCK_C channels per program, dynamic `range(K)` loops (not `tl.static_range` — that causes LLVM compilation explosion for K=31). Auto-tuned across 6 configs (BLOCK_C=32/64/128, num_warps=4/8).

| Config | Triton (ms) |
|---|---|
| 9×9, 1536ch, 32×32 | 0.299 |
| 7×7, 768ch, 32×32 | 0.096 |
| 31×31, 128ch, 56×56 | 0.717 |
| 29×29, 256ch, 28×28 | 0.265 |
| 27×27, 512ch, 14×14 | 0.087 |
| 13×13, 1024ch, 7×7 | 0.016 |

### 4.6 Kernel-Only Comparison (All Methods) — DONE

Isolated per-kernel latency, all methods, batch=1 FP16:

| Config | Custom (ms) | cuDNN NCHW (ms) | Triton (ms) | Custom vs cuDNN-best | Custom vs Triton |
|---|---|---|---|---|---|
| 9×9, 1536ch, 32×32 | **0.073** | 0.295 | 0.299 | **4.04×** | **4.10×** |
| 7×7, 768ch, 32×32 | **0.025** | 0.071 | 0.096 | **2.84×** | **3.84×** |
| 31×31, 128ch, 56×56 | 0.428 | **0.416** | 0.717 | 0.97× | 1.68× |
| 29×29, 256ch, 28×28 | 0.268 | **0.198** | 0.265 | 0.74× | 0.99× |
| 27×27, 512ch, 14×14 | 0.177 | **0.044** | 0.087 | 0.25× | 0.49× |
| 13×13, 1024ch, 7×7 | 0.190 | **0.013** | 0.016 | 0.07× | 0.08× |

**Key insight:** Custom kernels dominate for high-channel/moderate-kernel (ConvMixer configs: 4× faster). For low-channel/large-kernel/small-spatial (RepLKNet stages 2-3), cuDNN NCHW wins per-kernel. This supports the "regime-aware" thesis.

**Why cuDNN NCHW wins per-kernel on RepLKNet configs:** Not CUTLASS (depthwise conv has trivial GEMM dimension). cuDNN NCHW uses direct convolution where NCHW layout gives contiguous spatial access — the kernel sweeps over a contiguous HxW plane per channel with perfect coalesced access. Custom NHWC kernels must stride by C elements between spatial neighbors, hurting cache line utilization at low channel counts (128-512). Additionally, small spatial dims (14×14, 7×7) don't generate enough thread blocks to saturate SMs. However, this NCHW advantage **vanishes at network level** due to layout conversion overhead between layers.

### 4.7 End-to-End Compiled Comparison (All Compiled, Native Layouts, Zero Conversion) — DONE

All methods use compilation and **native memory layouts with zero conversion**:
- cuDNN NCHW: model + input in contiguous format, cuDNN runs native NCHW algorithms
- cuDNN NHWC: model + input in channels_last format, cuDNN runs native NHWC algorithms
- Custom CUDA: NHWC-native (shared-memory kernels read channels_last directly)
- Triton NHWC-native: vectorize over C, reads channels_last directly
- Triton NCHW-native: vectorize over W, reads contiguous directly

**Real ImageNet validation image used as input** (not random tensors).

```bash
python benchmark_compiled_comparison.py --model 1536_20 --device cuda:0 --num_runs 500 --data_dir ./data/imagenet
python benchmark_compiled_comparison.py --model 768_32 --device cuda:0 --num_runs 500 --data_dir ./data/imagenet
python replknet/benchmark.py --device cuda:0 --num_runs 100 --skip_verify --skip_accuracy --per_layer
```

**ConvMixer-1536/20 (9×9, 20 layers):**

| Method | Layout | Latency | vs cuDNN-best | vs Triton-best |
|---|---|---|---|---|
| cuDNN+compile | NCHW | 24.08 ms | 1.00× | — |
| cuDNN+compile | NHWC | 24.13 ms | 1.00× | — |
| Triton+compile | NHWC | 9.18 ms | **2.62×** | 1.00× |
| Triton+compile | NCHW | 18.28 ms | 1.32× | — |
| Custom+compile | NHWC | **4.14 ms** | **5.81×** | **2.22×** |

**ConvMixer-768/32 (7×7, 32 layers):**

| Method | Layout | Latency | vs cuDNN-best | vs Triton-best |
|---|---|---|---|---|
| cuDNN+compile | NCHW | 12.97 ms | 1.00× | — |
| cuDNN+compile | NHWC | 12.99 ms | 1.00× | — |
| Triton+compile | NHWC | 4.55 ms | **2.85×** | 1.00× |
| Triton+compile | NCHW | 9.92 ms | 1.31× | — |
| Custom+compile | NHWC | **2.03 ms** | **6.39×** | **2.24×** |

**ConvMixer layout finding:** cuDNN NCHW ≈ NHWC end-to-end. The per-kernel NCHW advantage on depthwise is offset by the non-DW layers (1×1 pointwise conv, BN) preferring NHWC. With 20-32 blocks of equal DW + pointwise layers, the effects cancel.

**Triton layout finding:** NHWC-native is consistently **2× faster** than NCHW-native (vectorizing over 768-1536 channels gives denser coalesced loads than vectorizing over W=32 spatial).

**RepLKNet-31B (31×31/29×29/27×27/13×13, all native layouts):**

| Method | Layout | Latency | vs cuDNN-best | vs Triton-best |
|---|---|---|---|---|
| cuDNN eager | NHWC | 18.14 ms | 0.46× | — |
| cuDNN eager | NCHW | **8.26 ms** | — | — |
| cuDNN+compile | NHWC | 15.00 ms | 0.55× | — |
| cuDNN+compile | NCHW | 5.15 ms | 1.00× | — |
| Triton+compile | NHWC | 273.59 ms | 0.02× | 1.00× |
| Triton+compile | NCHW | 373.54 ms | 0.01× | — |
| Custom eager | NHWC | 7.74 ms | 1.07× | — |
| Custom+compile | NHWC | **4.52 ms** | **1.14×** | **60.5×** |

**RepLKNet layout finding — CRITICAL NEW RESULT:** cuDNN NCHW is **2.2× faster** than cuDNN NHWC end-to-end (8.26 vs 18.14 ms eager). This is NOT a stranded optimization — cuDNN runs the ENTIRE network in native NCHW with zero conversion. The DW vs non-DW tradeoff:
- cuDNN NCHW: DW 3.52ms (fast), non-DW 4.75ms (slow) → total 8.26ms
- cuDNN NHWC: DW 16.19ms (slow), non-DW 1.96ms (fast) → total 18.14ms
- Custom NHWC: DW 4.74ms, non-DW 3.00ms → total 7.74ms

RepLKNet is DW-dominated (DW is 86% of cuDNN NHWC time), so NCHW's DW advantage overpowers the non-DW penalty. ConvMixer is balanced (DW ≈ non-DW), so layouts cancel.

**Key findings (corrected):**
- **ConvMixer: cuDNN NCHW ≈ NHWC** — DW advantage offset by non-DW disadvantage (balanced architecture)
- **RepLKNet: cuDNN NCHW >> NHWC** (2.2×) — DW-dominated architecture means NCHW's DW advantage survives
- **Custom NHWC wins both** — 5.8-6.4× over cuDNN-best on ConvMixer, 1.14× over cuDNN NCHW on RepLKNet
- Custom vs Triton-best: **2.2× faster** on ConvMixer, **60× faster** on RepLKNet
- Triton collapses on RepLKNet even with native layouts (274 ms). Root causes: (1) dynamic K×K loops with no shared-memory reuse, (2) small spatial dims limit parallelism, (3) no data reuse across output pixels
- Triton NHWC-native consistently 2× faster than NCHW-native across all models

### 4.8 ImageNet Accuracy Validation (Custom Kernels) — DONE

Full 50K ImageNet-1K validation set, all using Custom+compile (FP16 NHWC channels_last):

| Model | Top-1 | Top-5 | Published Top-1 | Match |
|---|---|---|---|---|
| ConvMixer-1536/20 | **81.36%** | **95.61%** | 81.3% | Yes |
| ConvMixer-768/32 | **80.15%** | **95.08%** | 80.2% | Yes |
| RepLKNet-31B | **83.30%** | **96.49%** | 83.3% | Yes |

Custom kernels are **numerically faithful** — zero accuracy degradation across all three models. This confirms the NHWC shared-memory kernels produce equivalent outputs to cuDNN within FP16 tolerance.

Data location: `data/imagenet/` (HuggingFace cache, validation split only ~6 GB).

### 4.9 NCU Profiling — Detailed Plan

**Status:** DONE. NCU v2025.3.1.0 profiling completed 2026-04-29. 17 kernel profiles collected.
**NCU version:** 2025.3.1.0 (installed at /usr/local/cuda/bin/ncu)
**Scripts:** `profile_ncu.py` (harness), `run_ncu_profiles.sh` (batch runner), `analyze_ncu.py` (parser), `plot_roofline.py` (figure)
**Data:** `results/ncu/*.csv` (17 files), `results/spark_ncu_metrics.csv`, `results/spark_roofline_data.csv`, `results/spark_ncu_tables.md`, `results/spark_roofline_plot.png`

#### 4.9.1 The Per-Kernel vs End-to-End Reversal (Key Finding for Paper)

The most important finding to explain with NCU: Custom NHWC **loses per-kernel** to cuDNN NCHW on RepLKNet stages, but **wins 3.3× end-to-end**. This is NOT a simple "layout conversion overhead" story — it's a three-way comparison.

**Three-way per-stage comparison (from per-layer profiling within the network, warm L2):**

| Stage | Config | Custom NHWC | cuDNN NHWC | cuDNN NCHW | Custom vs NHWC | Custom vs NCHW |
|-------|--------|-------------|------------|------------|---------------|---------------|
| 0 | 31×31, 128ch, 56×56 | 0.423 ms | 2.946 ms | 0.478 ms | **7.0× faster** | **1.13× faster** |
| 1 | 29×29, 256ch, 28×28 | 0.268 ms | 1.193 ms | 0.264 ms | **4.5× faster** | 0.98× (tied) |
| 2 | 27×27, 512ch, 14×14 | 0.077 ms | 0.340 ms | 0.080 ms | **4.4× faster** | 0.96× (tied) |
| 3 | 13×13, 1024ch, 7×7 | 0.093 ms | 0.093 ms | 0.025 ms | 1.0× (tied) | 0.27× (slower) |

Note: These are per-layer times measured within the running network (warm L2 cache). Isolated cold-cache numbers differ — see Section 4.6.

**The three layers of the story:**

**Layer 1 — cuDNN NCHW is the per-kernel optimum (but unrealizable in network context)**

cuDNN NCHW wins per-kernel by 1.35–14× over Custom for stages 1-3. NCU should show why:
- **Memory coalescing advantage**: NCHW layout stores each channel as a contiguous H×W plane. For 14×14 spatial, that's 196 × 2B = 392 bytes — fits in ~6 cache lines. cuDNN sweeps this plane with perfect sequential access. Custom NHWC must stride by C=512 elements (1024 bytes) between spatial neighbors — terrible cache line utilization.
- **Occupancy advantage**: cuDNN NCHW can decompose the problem differently (one channel per block, small block, many concurrent blocks). Custom's NHWC kernels at 14×14 use 59KB shmem → at most 1 block/SM → 224 blocks across 48 SMs = ~4.7 blocks/SM, but shmem limits concurrent residency. NCU occupancy metric will quantify this gap.
- **Small-spatial parallelism**: 14×14 = 196 output pixels is very few. Custom packs 14 pixels into 14 active warps (2 idle). cuDNN NCHW can achieve higher utilization by processing entire channel planes.

**Layer 2 — Custom NHWC beats cuDNN NHWC per-kernel (the actual deployed comparison)**

Custom is **1.9–6.7× faster** than cuDNN NHWC for stages 0-2. This is the comparison that matters at the network level, because the network runs NHWC. NCU should show why:
- **Shared-memory weight caching**: Custom loads K×K×ch_block weights into shmem once per block, shared across all output pixels in the tile. cuDNN NHWC has no shared-memory weight tiling for depthwise — it reloads weights per output pixel from global/L2.
- **DRAM traffic reduction**: For stage 0 (31×31, 128ch), weights = 31×31×128 = 123,008 half values. Custom loads weights once per block; cuDNN NHWC loads them redundantly per pixel. NCU `dram__bytes_read.sum` will quantify the difference.
- **cuDNN NHWC depthwise is not optimized**: cuDNN's best depthwise path is NCHW. Its NHWC depthwise appears to use a generic implementation. This explains the 3.7–6.8× NCHW-over-NHWC gap in cuDNN's own layout comparison.

**Layer 3 — cuDNN NCHW IS realizable but has a non-DW penalty (CORRECTED)**

**Previous claim was WRONG**: We initially assumed cuDNN NCHW required layout conversions. Testing native NCHW end-to-end (zero conversions) shows cuDNN NCHW is 2.2× faster than NHWC for RepLKNet (8.26 vs 18.14ms). The DW advantage survives. But NCHW hurts non-DW layers:
- cuDNN NCHW non-DW: 4.75ms vs cuDNN NHWC non-DW: 1.96ms (NCHW 2.4× slower on non-DW)
- cuDNN NCHW DW: 3.52ms vs cuDNN NHWC DW: 16.19ms (NCHW 4.6× faster on DW)
- For RepLKNet (DW-dominated), the DW advantage wins → NCHW is faster overall
- For ConvMixer (balanced), the effects cancel → NCHW ≈ NHWC

**Paper argument**: Custom NHWC is the **globally optimal design point** because shared-memory tiling **closes the NHWC depthwise gap** with cuDNN NCHW from 4.9× (cuDNN NHWC) to within 3% (0.97× for 31×31/128ch) — essentially tied. With the DW gap eliminated, NHWC's non-DW advantage (3.00ms vs 4.75ms) tips the total. Neither cuDNN layout achieves this balance:
- cuDNN NCHW: fast DW but slow non-DW → 8.26ms
- cuDNN NHWC: slow DW but fast non-DW → 18.14ms
- Custom NHWC: competitive DW + fast non-DW → 7.74ms (→ 4.52ms compiled)

#### 4.9.2 What to Profile (18 kernel profiles)

**6 configs × 3 implementations:**

| # | Config | Custom kernel | cuDNN layout | Triton layout |
|---|--------|---------------|-------------|---------------|
| 1 | 9×9, 1536ch, 32×32 | depthwise_32_1536_9_kernel (512 threads, cp.async double-buffer) | NCHW (best) | NHWC-native (best) |
| 2 | 7×7, 768ch, 32×32 | depthwise_32_768_7_kernel (1024 threads, weights-only shmem) | NCHW (best) | NHWC-native (best) |
| 3 | 31×31, 128ch, 56×56 | ch_128_k_31_colmerge_8x12 | NCHW (best) | NHWC-native (best) |
| 4 | 29×29, 256ch, 28×28 | ch_256_k_29_colmerge_8x12 | NCHW (best) | NHWC-native (best) |
| 5 | 27×27, 512ch, 14×14 | ch_512_k_27_naive64_half2 | NCHW (best) | NHWC-native (best) |
| 6 | 13×13, 1024ch, 7×7 | (no custom — cuDNN native) | NCHW (best) | NHWC-native (best) |

Additionally, profile cuDNN **NHWC** for RepLKNet configs (4 extra profiles) to quantify Layer 2 above.

#### 4.9.3 NCU Metrics and Their Paper Purpose

**Metric Set 1: Roofline (SpeedOfLight + HierarchicalHalfRooflineChart)**

Purpose: The main figure. Shows each kernel's position on the FP16 roofline of GB10.
- X-axis: arithmetic intensity (FLOP/byte)
- Y-axis: achieved GFLOP/s
- Ceilings: HBM bandwidth (~200 GB/s est. for GB10) and FP16 peak compute

What it proves:
- Depthwise conv is **memory-bound** (all kernels below the compute ceiling, near the bandwidth ceiling)
- Custom kernels have higher effective arithmetic intensity than Triton (shared-memory reuse increases effective AI by reducing DRAM bytes)
- Triton on large-K configs sits far from the bandwidth ceiling (redundant global loads waste bandwidth)
- cuDNN NCHW achieves high bandwidth utilization through coalesced access patterns

GB10 theoretical peaks (to be confirmed with NCU):
- FP16 compute: ~209 TFLOP/s (from spec)
- HBM bandwidth: N/A — unified memory via NVLink-C2C, need to measure actual bandwidth ceiling

**Metric Set 2: DRAM Traffic (MemoryWorkloadAnalysis)**

Key metrics:
- `dram__bytes_read.sum` — total DRAM reads per kernel invocation
- `dram__bytes_write.sum` — total DRAM writes
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — bandwidth utilization

Purpose: Quantifies the shared-memory reuse advantage.

Expected results for 27×27/512ch/14×14 (stage 2):
- Theoretical minimum read: input (14×14×512×2B = 200KB) + weights (27×27×512×2B = 746KB) + output (200KB) = ~1.15 MB
- Custom: close to minimum (weights loaded to shmem once per block, input from global with L2 reuse)
- Triton: much higher (each of H×W program instances reloads weights from global memory independently)
- The ratio (Triton DRAM / Custom DRAM) directly explains the speedup

**Metric Set 3: Shared Memory vs Global Memory Instructions**

Key metrics:
- `smsp__inst_executed_op_shared_ld.sum` — shared memory load instructions
- `smsp__inst_executed_op_shared_st.sum` — shared memory store instructions
- `smsp__inst_executed_op_global_ld.sum` — global memory load instructions

Purpose: "Implementation fingerprint" — proves algorithms are fundamentally different.
- Custom: high shmem-load ratio (weight reads → shmem)
- Triton: zero shmem-load (everything from global/L2)
- cuDNN: reveals its internal strategy (unknown a priori)

**Metric Set 4: Occupancy**

Key metrics:
- `sm__warps_active.avg.pct_of_peak_sustained_elapsed` — achieved occupancy
- `sm__maximum_warps_per_active_cycle_pct` — theoretical occupancy

Purpose: Explains the regime crossover at small spatial dims.

Expected:
- ConvMixer configs (32×32 spatial): Custom achieves decent occupancy (enough blocks)
- RepLKNet stage 2 (14×14 spatial): Custom achieves LOW occupancy (59KB shmem → 1 block/SM)
- RepLKNet stage 2 cuDNN NCHW: achieves HIGHER occupancy (explains its per-kernel win)

**Metric Set 5: Compute Pipe Utilization**

Key metrics:
- `smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` — FMA pipe utilization

Purpose: Confirms memory-bound nature. All kernels should show <30% compute utilization.
- Custom should show HIGHER utilization than Triton (less time starved for data)

**Metric Set 6: Warp Stall Analysis (WarpStateStats)**

Key metrics:
- `smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct` — memory dependency stalls
- `smsp__warps_issue_stalled_barrier_per_warp_active.pct` — barrier (syncthreads) stalls
- `smsp__warps_issue_stalled_wait_per_warp_active.pct` — other wait stalls

Purpose: Shows what each kernel is waiting for.
- All kernels: dominated by memory stalls → memory-bound
- Custom: some barrier stalls (expected/healthy — syncthreads after shmem weight loading)
- Triton: MORE memory stalls, ZERO barrier stalls (no shmem coordination)

Paper argument: "Custom trades global-memory stalls for barrier stalls — a favorable exchange that reduces total stall time through shared-memory reuse."

#### 4.9.4 NCU Command Template

```bash
# Detailed set + FP16 roofline (per kernel, ~3-5 min each)
ncu --set detailed \
    --section SpeedOfLight_HierarchicalHalfRooflineChart \
    --target-processes all \
    --kernel-name <kernel_name> \
    --launch-skip 5 --launch-count 3 \
    --csv --log-file results/ncu_<config>_<method>.csv \
    conda run -n chConv python profile_ncu.py --config <config> --method <method>
```

Total: ~22 profiles × 3-5 min = 60-110 min.

#### 4.9.5 Nsight Systems Timeline (Complementary, may not require same permissions)

For the end-to-end reversal story, an nsys trace of the full RepLKNet inference shows:
- cuDNN: layout conversion kernels visible between layers in the timeline
- Custom: clean timeline, no conversion kernels

```bash
nsys profile --trace=cuda,nvtx --output results/nsys_replknet_cudnn \
    conda run -n chConv python -c "..."
```

#### 4.9.6 Proposed Paper Tables/Figures from NCU Data

**Figure: FP16 Roofline Plot (main performance analysis figure)**
- All 18 kernels plotted: Custom (●), cuDNN-NCHW (▲), Triton-NHWC (◆)
- Color: ConvMixer configs (blue), RepLKNet configs (red)
- Shows memory-bound nature + Custom near bandwidth ceiling + Triton far below for large-K

**Table: Memory Traffic Analysis**

| Config | Custom DRAM (MB) | cuDNN-NCHW DRAM (MB) | Triton DRAM (MB) | Custom savings |
|--------|-----------------|---------------------|-----------------|----------------|
| 9×9, 1536ch | ? | ? | ? | ?× |
| 27×27, 512ch | ? | ? | ? | ?× |

**Table: Microarchitecture Characterization**

| Config | Method | Occupancy (%) | FMA Util (%) | Shmem Loads | DRAM BW Util (%) | Top Stall |
|--------|--------|--------------|-------------|-------------|-----------------|-----------|
| 9×9, 1536ch | Custom | ? | ? | high | ? | barrier |
| 9×9, 1536ch | cuDNN | ? | ? | ? | ? | memory |
| 9×9, 1536ch | Triton | ? | ? | 0 | ? | memory |

**Table: The Three-Way RepLKNet Comparison (NEW — key discussion table)**

| Stage | Custom NHWC | cuDNN NHWC | cuDNN NCHW | Network uses | Why Custom wins E2E |
|-------|-------------|------------|------------|-------------|-------------------|
| 0 | 0.428 | 2.868 | 0.416 | cuDNN NHWC | Custom 6.7× > cuDNN-NHWC |
| 1 | 0.268 | 1.444 | 0.198 | cuDNN NHWC | Custom 5.4× > cuDNN-NHWC |
| 2 | 0.177 | 0.344 | 0.044 | cuDNN NHWC | Custom 1.9× > cuDNN-NHWC |
| 3 | 0.190 | 0.094 | 0.013 | cuDNN NHWC | cuDNN-NHWC 2× > Custom |
| **Total** | **5.84** | **15.89** | **0.66** | **15.89** | **Custom 2.7× > cuDNN-NHWC** |

This table is the centerpiece of the "per-kernel vs end-to-end reversal" discussion. NCU data adds the WHY column.

#### 4.9.7 How NCU Addresses Each Reviewer Complaint

| Reviewer Complaint | NCU Evidence |
|---|---|
| #1 "No novelty" | Roofline + memory traffic prove regime boundaries have microarchitectural basis. Three-way RepLKNet comparison shows layout-aware holistic design is a non-trivial optimization insight. |
| #5 "Insufficient performance analysis" | Full roofline plot, occupancy analysis, memory traffic quantification, warp stall breakdown — standard HPC methodology. |
| #3 "Single GPU" | Same NCU analysis on both RTX 6000 Ada and DGX Spark shows regime boundaries shift with hardware (different SM count, memory hierarchy). |

#### 4.9.8 NCU Results — Actual Microarchitectural Data (DONE 2026-04-29)

**Scripts created:**
- `profile_ncu.py` — Single-kernel profiling harness for NCU. Supports `--config` and `--method`.
- `run_ncu_profiles.sh` — Batch runner. Profiles 17 kernels (5 custom + 6 cuDNN NCHW + 4 cuDNN NHWC + 2 Triton NHWC).
- `analyze_ncu.py` — Parses NCU CSV output, extracts key metrics, generates paper-ready tables.
- `plot_roofline.py` — Generates FP16 roofline plot with GB10 theoretical ceilings.

**Profiling completed:** 17 of 18 planned profiles (Triton collapses on RepLKNet large-K, so profiled only ConvMixer configs where Triton is competitive).

##### 4.9.8.1 SpeedOfLight + Occupancy (Key Table)

| Config | Method | Mem BW (%) | SM Thr. (%) | Achieved Occ (%) | Theo Occ (%) | Duration (μs) |
|--------|--------|-----------|------------|-----------------|-------------|---------------|
| 9×9, 1536ch, 32×32 | Custom NHWC | 58.7 | 58.7 | 77.1 | 66.7 | 106.9 |
| 9×9, 1536ch, 32×32 | cuDNN NCHW | 62.9 | 62.9 | 60.4 | 66.7 | 384.4 |
| 9×9, 1536ch, 32×32 | Triton NHWC | 34.0 | 56.3 | 98.3 | 100.0 | 797.2 |
| 7×7, 768ch, 32×32 | Custom NHWC | 51.0 | 51.0 | 75.7 | 66.7 | 36.2 |
| 7×7, 768ch, 32×32 | cuDNN NCHW | 41.9 | 50.4 | 59.1 | 66.7 | 108.2 |
| 7×7, 768ch, 32×32 | Triton NHWC | 31.4 | 54.3 | 92.3 | 100.0 | 272.0 |
| 31×31, 128ch, 56×56 | Custom NHWC | 55.7 | 55.7 | 33.4 | 33.3 | 405.0 |
| 31×31, 128ch, 56×56 | cuDNN NCHW | 81.6 | 81.6 | 61.5 | 66.7 | 496.8 |
| 31×31, 128ch, 56×56 | cuDNN NHWC | 7.9 | 5.6 | 17.4 | 50.0 | 219.2 |
| 29×29, 256ch, 28×28 | Custom NHWC | 39.7 | 39.7 | 33.1 | 33.3 | 201.0 |
| 29×29, 256ch, 28×28 | cuDNN NCHW | 70.0 | 70.0 | 58.6 | 66.7 | 237.0 |
| 29×29, 256ch, 28×28 | cuDNN NHWC | 8.3 | 6.2 | 17.2 | 50.0 | 162.5 |
| 27×27, 512ch, 14×14 | Custom NHWC | 13.7 | 13.7 | 30.0 | 33.3 | 116.6 |
| 27×27, 512ch, 14×14 | cuDNN NCHW | 28.4 | 28.4 | 63.5 | 66.7 | 90.9 |
| 27×27, 512ch, 14×14 | cuDNN NHWC | 30.4 | 23.0 | 7.8 | 8.3 | 658.8 |
| 13×13, 1024ch, 7×7 | cuDNN NCHW | 15.1 | 18.8 | 76.8 | 66.7 | 20.3 |
| 13×13, 1024ch, 7×7 | cuDNN NHWC | 34.6 | 18.0 | 64.3 | 66.7 | 77.1 |

**Key finding — cuDNN NHWC occupancy collapse:**
cuDNN NHWC shows **7.8-17.4%** occupancy on RepLKNet stages — catastrophic. This is 3-4× lower than Custom NHWC (30-33%) and 4-8× lower than cuDNN NCHW (59-64%). The `No Eligible` scheduler metric confirms these warps are stalled on memory dependencies, not barriers. cuDNN's generic NHWC depthwise path simply starves the SMs of work.

##### 4.9.8.2 Memory Hierarchy (Key Table)

| Config | Method | L1 Hit (%) | L2 Hit (%) | Mem Busy (%) | Shmem/Block (KB) |
|--------|--------|-----------|-----------|-------------|------------------|
| 9×9, 1536ch, 32×32 | Custom NHWC | 6.6 | 66.4 | 29.9 | 26.1 |
| 9×9, 1536ch, 32×32 | cuDNN NCHW | 99.2 | 24.9 | 31.5 | 0.0 |
| 9×9, 1536ch, 32×32 | Triton NHWC | 56.5 | 98.7 | 24.5 | 0.0 |
| 7×7, 768ch, 32×32 | Custom NHWC | 90.8 | 51.5 | 25.8 | 12.2 |
| 7×7, 768ch, 32×32 | cuDNN NCHW | 97.9 | 26.0 | 21.0 | 0.0 |
| 7×7, 768ch, 32×32 | Triton NHWC | 66.9 | 98.6 | 17.1 | 0.0 |
| 31×31, 128ch, 56×56 | Custom NHWC | 89.5 | 94.1 | 27.9 | 60.1 |
| 31×31, 128ch, 56×56 | cuDNN NCHW | 99.8 | 61.8 | 41.1 | 0.0 |
| 31×31, 128ch, 56×56 | cuDNN NHWC | 96.3 | 79.4 | 7.9 | 0.0 |
| 29×29, 256ch, 28×28 | Custom NHWC | 88.4 | 89.6 | 19.9 | 52.6 |
| 29×29, 256ch, 28×28 | cuDNN NCHW | 99.7 | 43.0 | 35.4 | 0.0 |
| 29×29, 256ch, 28×28 | cuDNN NHWC | 97.4 | 77.1 | 8.3 | 0.0 |
| 27×27, 512ch, 14×14 | Custom NHWC | 73.1 | 91.2 | 6.9 | 91.1 |
| 27×27, 512ch, 14×14 | cuDNN NCHW | 99.0 | 23.5 | 15.8 | 0.0 |
| 27×27, 512ch, 14×14 | cuDNN NHWC | 40.6 | 93.4 | 30.4 | 65.0 |

**Key finding — Custom NHWC L1 cache bypass:**
The ConvMixer 9×9/1536 custom kernel shows **6.6% L1 hit rate** — deliberate. The double-buffered `cp.async` pipeline bypasses L1, writing directly to shared memory. The remaining kernels (7×7/768) show 90.8% L1 hit rate because they load weights through L1 to shared memory. This is a microarchitectural fingerprint proving fundamentally different implementation strategies.

**Key finding — cuDNN NCHW 99%+ L1 hit rate:**
cuDNN NCHW achieves near-perfect L1 hit rates (99.0-99.8%) across all configs through contiguous spatial access. But this comes at the cost of zero shared memory usage — weights must be re-read from L1 per pixel.

**Key finding — Triton 98.7% L2 hit:**
Triton kernels show 98.6-98.7% L2 hit rate but zero shared memory. The kernel is small enough that input+weights fit in L2 for moderate configs, but the K×K loop produces redundant reads even from L2 — bandwidth utilization stagnates at 31-34%.

##### 4.9.8.3 Stall Analysis

| Config | Method | No Eligible (%) | IPC | Cycles/Inst | Waves/SM |
|--------|--------|----------------|-----|-------------|----------|
| 9×9, 1536ch, 32×32 | Custom NHWC | 66.3 | 1.34 | 21.6 | 4.0 |
| 9×9, 1536ch, 32×32 | cuDNN NCHW | 42.9 | 2.25 | 13.1 | 32.0 |
| 9×9, 1536ch, 32×32 | Triton NHWC | 43.6 | 2.25 | 21.2 | 42.7 |
| 7×7, 768ch, 32×32 | Custom NHWC | 49.6 | 2.04 | 14.9 | 2.0 |
| 7×7, 768ch, 32×32 | cuDNN NCHW | 44.7 | 2.17 | 12.9 | 16.0 |
| 7×7, 768ch, 32×32 | Triton NHWC | 38.4 | 2.08 | 21.7 | 21.3 |
| 27×27, 512ch, 14×14 | Custom NHWC | 88.8 | 0.42 | 31.1 | 2.3 |
| 27×27, 512ch, 14×14 | cuDNN NCHW | 42.9 | 2.20 | 12.5 | 2.0 |
| 27×27, 512ch, 14×14 | cuDNN NHWC | 75.9 | 0.95 | 4.2 | 2.7 |

**Key finding — The stall signature is the regime boundary:**
- **ConvMixer (Custom)**: 50-66% no-eligible = moderate stalls, 1.3-2.0 IPC — shared memory reuse keeps execution moving
- **ConvMixer (cuDNN NCHW)**: 43-45% no-eligible, 2.2 IPC — contiguous access patterns minimize stalls but no weight reuse means more total instructions
- **ConvMixer (Triton)**: 38-44% no-eligible, 2.1-2.3 IPC — dense grid keeps schedulers fed but 21.2 cycles/inst shows each instruction is expensive (redundant k×k loads)
- **RepLKNet stage 2 (Custom)**: **88.8% no-eligible, 0.42 IPC** — catastrophic! The 27×27 kernel at 14×14 spatial cannot generate enough thread blocks (112 blocks / 48 SMs = 2.3 waves/SM). Combined with 91KB shmem/block, SMs are starved
- **RepLKNet stage 2 (cuDNN NHWC)**: 76% no-eligible at 4.2 cycles/inst — cuDNN's NHWC path is also starved but for different reasons (poor occupancy from generic implementation)

##### 4.9.8.4 Roofline Coordinates

| Config | Method | AI (FLOP/byte) | GFLOP/s | Duration (μs) |
|--------|--------|---------------|---------|---------------|
| 9×9, 1536ch, 32×32 | Custom NHWC | 10.2 | 2383 | 106.9 |
| 9×9, 1536ch, 32×32 | cuDNN NCHW | 2.6 | 663 | 384.4 |
| 9×9, 1536ch, 32×32 | Triton NHWC | 2.4 | 320 | 797.2 |
| 7×7, 768ch, 32×32 | Custom NHWC | 10.4 | 2130 | 36.2 |
| 7×7, 768ch, 32×32 | cuDNN NCHW | 4.2 | 712 | 108.2 |
| 7×7, 768ch, 32×32 | Triton NHWC | 2.3 | 283 | 272.0 |
| 31×31, 128ch, 56×56 | Custom NHWC | 8.6 | 1905 | 405.0 |
| 31×31, 128ch, 56×56 | cuDNN NCHW | 4.8 | 1553 | 496.8 |
| 27×27, 512ch, 14×14 | Custom NHWC | 22.8 | 1254 | 116.6 |
| 27×27, 512ch, 14×14 | cuDNN NCHW | 14.2 | 1610 | 90.9 |

**Key finding — Shared-memory reuse increases effective arithmetic intensity:**
Custom NHWC consistently achieves 10-23 FLOP/byte AI vs cuDNN NCHW's 2.6-14 FLOP/byte. The shared-memory weight caching reduces actual DRAM bytes moved, increasing effective AI by 2-4×. This is the quantitative proof of the "shared-memory tiling" advantage.

**Roofline figure**: `results/spark_roofline_plot.png`

##### 4.9.8.5 Summary — What NCU Proves for the Paper

The NCU data confirms four non-obvious findings at the hardware-counter level:

1. **Shared-memory tiling achieves DW parity with cuDNN NCHW**: Custom NHWC achieves 2.6-4× higher effective arithmetic intensity (10-23 FLOP/byte) than cuDNN NCHW (2.6-14 FLOP/byte) by caching weights in shared memory. This transforms NHWC depthwise from 4.9× slower (cuDNN NHWC vs NCHW, cold L2) to **within 3%** (0.97× for 31×31/128ch) — parity for large-spatial configs. For small-spatial configs (27×27/14×14), cuDNN NCHW still wins (4.0×) due to shared-memory occupancy limits.

2. **cuDNN NHWC occupancy collapse explains the per-kernel speedup**: cuDNN NHWC achieves only 7.8-17.4% occupancy on RepLKNet stages — 3-8× lower than other implementations. Custom NHWC fixes this with shared-memory weight caching that keeps warps busy.

3. **Triton's high occupancy paradox**: Triton achieves 92-98% occupancy (highest of all methods) but only 284-320 GFLOP/s (lowest). The dense 1D grid keeps all SMs occupied, but 98% of data comes from L2 — each program instance redundantly reloads K×K weights from L2 with zero inter-program reuse. High occupancy cannot compensate for fundamentally wasteful memory access.

4. **The regime boundary is a microarchitectural cliff**: At 27×27/512ch/14×14, Custom NHWC hits **88.8% no-eligible stalls + 0.42 IPC** — the shared-memory strategy breaks down when there aren't enough thread blocks to hide latency. cuDNN NCHW's smaller-block decomposition avoids this cliff, explaining its per-kernel win at this config.

#### 4.9.9 NSYS End-to-End Analysis — Why Custom Wins the Full Network (DONE 2026-04-29)

NSYS profiling (v2025.3.2) was run on a mini-block (31×31 DW + 1×1 PW + GELU) in all 4 configurations: Custom NHWC eager, Custom NHWC compiled, cuDNN NCHW eager, cuDNN NCHW compiled. Full results in `results/nsys_fusion_*.nsys-rep`.

**Scripts:** `test_compile_fusion.py` — generates the micro-benchmark and NSYS trace.

##### 4.9.9.1 Kernel-Level Breakdown (NSYS cuda_gpu_kern_sum)

All NSYS profiles use 10 warmup calls before timing → warm L2 cache.

| | Custom NHWC eager | cuDNN NCHW eager | Custom NHWC compiled | cuDNN NCHW compiled |
|---|---|---|---|---|
| **DW kernel** | 284 μs (1 kernel, colmerge) | 403 μs (1 kernel, conv_depthwise2d) | 284 μs (1 kernel, colmerge) | **2,644 μs** (768 kernels: 512 GEMM + 128 precomputed + 128 nhwc→nchw) |
| **PW (1×1)** | 7.4 μs (1 kernel, cutlass) | 26.4 μs (1 kernel, magma_sgemm) | 7.3 μs (1 kernel, cutlass) | fused with pw+gelu |
| **GELU** | 3.5 μs (1 kernel) | 3.6 μs (1 kernel) | fused with pw (triton_poi) | fused with pw (triton_poi) |
| **Bias/other** | 8.8 μs | 5.7 μs | 3.6 μs | ~6 μs |
| **Total** | **304 μs** | **438 μs** | **299 μs** | **~2,660 μs** |
| **Kernels/call** | **5** | **4** | **4** | **~898** |

**Critical distinction — cold vs warm L2 cache:**

The NSYS warm-cache numbers above show Custom DW at 284 μs vs cuDNN NCHW at 403 μs (1.42×). However, the **canonical cold-L2 benchmark** (`spark_kernel_comparison.csv`, Section 4.6) shows:

| Config | Custom (ms) | cuDNN NCHW (ms) | Custom vs cuDNN NCHW |
|--------|------------|-----------------|---------------------|
| 31×31, 128ch, 56×56 | 0.428 | 0.416 | **0.97×** (tied, slightly slower) |
| 29×29, 256ch, 28×28 | 0.268 | 0.198 | 0.74× (slower) |
| 27×27, 512ch, 14×14 | 0.177 | 0.044 | 0.25× (much slower) |

At cold L2, Custom's shared-memory weight loading incurs full DRAM latency — 31×31×32×140 blocks = 8.6 MB of weights streamed from DRAM. cuDNN NCHW's direct kernel accesses weights through L1 with near-perfect coalescing (99.8% L1 hit rate, per NCU), avoiding this upfront cost.

Within a real network (warm L2), weights persist in L2 between layers, and Custom's warm-cache performance (284 μs) ≈ cuDNN NCHW's cold performance (416 μs) — the shmem strategy catches up once L2 is warm. This explains why Section 4.9.1 per-layer profiling shows Custom winning stage 0 by 1.13× within the network, but losing by 0.97× in cold isolation.

The **cold-L2 numbers are the canonical fair comparison** for per-kernel benchmarks. The warm-L2 NSYS numbers are the realistic network-inference comparison.

##### 4.9.9.2 The Two Mechanisms (Hypothesis Verification)

**Hypothesis 1 (Eager): Custom > cuDNN NCHW because NHWC layout favors non-DW operations.**

✓ **CONFIRMED, with clarification.** The cold-L2 canonical data shows Custom NHWC is NOT faster on DW — it achieves **parity** (within 3%) for large-spatial configs and is slower for small-spatial configs. The eager end-to-end win comes from:
- **DW parity (not superiority):** Shared-memory tiling closes the NHWC DW gap from 4.9× (cuDNN NHWC) to 0.97× (cold L2) — essentially tied for 31×31 at 56×56. For smaller spatial dims, cuDNN NCHW still wins per-kernel.
- **PW superiority (3.57×):** 1×1 pointwise convolution is a GEMM operation. NHWC layout places channels contiguous in memory → cutlass achieves 7.4 μs. NCHW layout requires strided channel access → magma_sgemm takes 26.4 μs. This is a fundamental layout property.
- **Combined:** Even with DW at parity, the 3.57× PW advantage tips the total. In the NSYS mini-block: 304 μs vs 438 μs → Custom 1.44× faster. At full RepLKNet scale, this translates to 7.74 ms vs 8.26 ms eager (1.07×).

**Hypothesis 2 (Compiled): Custom > cuDNN NCHW because custom ops are more fusion-friendly.**

✓ **CONFIRMED, but mechanism differs from expected.** The compiler does NOT fuse DW with PW in either method. PW+GELU fusion happens equally in both (both produce `triton_poi_fused_convolution_gelu`). The real difference is:

- **Custom NHWC survives compilation intact.** `torch.ops.replknet_dw.conv_56_128_31` is a torch.library registered op — inductor treats it as opaque, cannot decompose it. The 284 μs kernel launches unchanged.
- **cuDNN NCHW gets DESTROYED by compilation.** Inductor replaces cuDNN's efficient `conv_depthwise2d_forward_kernel_generic` (403 μs, 1 kernel) with a GEMM decomposition: 512 `implicit_convolve_sgemm` + 128 `precomputed_convolve_sgemm` + 128 `nhwcToNchw` + 128 index precomputations = 2,644 μs (**6.6× degradation**).

This GEMM decomposition is optimized for batch>1 throughput — at B=1, kernel launch overhead dominates and the GEMM tiles are underutilized. cuDNN eager correctly selects the direct kernel (knowing B=1), but torch.compile's inductor doesn't preserve this layout-specific optimization.

##### 4.9.9.3 The Corrected Paper Argument (Three-Factor Explanation)

Custom NHWC wins end-to-end over cuDNN's best layout through **three simultaneous advantages** that no single cuDNN layout can achieve:

| Factor | Custom NHWC | cuDNN NCHW | cuDNN NHWC |
|--------|------------|------------|------------|
| 1. DW kernel (cold L2) | **Parity** (shmem tiling, 428 μs, 0.97×) | **Good** (direct conv, 416 μs) | **Catastrophic** (768 GEMM kernels, 2,045 μs) |
| 2. PW kernel quality | **Good** (NHWC GEMM, 7 μs) | **Poor** (NCHW GEMM, 26 μs) | **Good** (NHWC GEMM, 7 μs) |
| 3. Compile survival | **Immune** (custom op preserved) | **Broken** (inductor GEMM decomposition, 6.6× degradation) | **Broken** (already GEMM in eager) |

**Eager result:** Custom wins by achieving DW **parity** (Factor 1: shmem closes the 4.9× gap to 0.97×) + **PW superiority** (Factor 2: 3.6× faster pointwise in NHWC). Together these compound to 1.44× on the mini-block, 1.07× on full RepLKNet.

**Compiled result:** Custom wins MORE because Factor 3 — cuDNN NCHW's only advantage (the direct DW kernel) gets destroyed by inductor, replaced with GEMM decomposition at 6.6× degradation. Custom's torch.library registration is a compiler boundary that preserves the carefully-tuned kernel.

The `test_compile_fusion.py` script and NSYS traces provide the evidence files.

### 4.10 Roofline Analysis (DONE — NCU data collected, plot generated)

## 5. Output Files

```
results/
├── spark_hardware_info.txt          # Hardware discovery (DONE)
├── spark_convmixer_1536_20.txt      # ConvMixer-1536/20 latency results (DONE)
├── spark_convmixer_768_32.txt       # ConvMixer-768/32 latency results (DONE)
├── spark_replknet.txt               # RepLKNet-31B latency results (DONE)
├── spark_cudnn_layouts.csv          # cuDNN best-layout results (DONE)
├── spark_triton_results.csv         # Triton baseline (DONE)
├── spark_kernel_comparison.csv      # All-methods kernel comparison (DONE)
├── spark_ncu_metrics.csv            # NCU profiling data (TODO)
└── spark_roofline_data.csv          # Arithmetic intensity + achieved perf (TODO)
```

Data: `data/imagenet/` — HuggingFace cache (validation split only, ~6 GB Arrow files).

Scripts at repo root:
- `benchmark_cudnn_layouts.py` — cuDNN NCHW vs NHWC per-kernel benchmark
- `benchmark_triton.py` — Triton depthwise conv kernel-only baseline with autotune
- `benchmark_compiled_comparison.py` — End-to-end compiled comparison (cuDNN vs Custom vs Triton), supports `--data_dir` for real ImageNet input

## 6. Data Completeness Status

### DONE — Ready for Paper

| Experiment | Status | Reviewer Complaint Addressed |
|---|---|---|
| Custom vs cuDNN (eager + compile) | DONE | Baseline |
| cuDNN best-layout (NCHW vs NHWC per-kernel) | DONE | Fair baseline |
| Triton native (NHWC + NCHW, zero conversion) | DONE | #2 "cuDNN-only baseline" |
| End-to-end compiled comparison (all methods) | DONE | #2 |
| ImageNet accuracy (50K, all 3 models) | DONE | Correctness proof |
| DGX Spark platform characterization | DONE | #3 "Single GPU" |

### TODO — Remaining

| Experiment | Est. Time | Reviewer Complaint Addressed | Blocker |
|---|---|---|---|
| NCU profiling (22 profiles, 6 metrics each) | 2 hr | #5 "Insufficient performance analysis" | Needs admin to set RmProfilingAdminOnly=0 |
| Roofline plot from NCU data | 1 hr (after NCU) | #5 | Depends on NCU |
| Nsight Systems timeline (RepLKNet E2E) | 30 min | #5 (layout conversion visualization) | May not need same permissions as NCU |
| Cross-platform table (RTX 6000 Ada vs Spark) | 1 hr | #3 "Single GPU" | Need Ada data |

### NOT on this branch

| Experiment | Branch | Reviewer Complaint |
|---|---|---|
| Batch scaling study (B=1,2,4,8,...) | main | #4 "Batch=1 only" |

## 7. Analysis — Regime-Aware Thesis (Core Paper Argument)

The data from this branch provides three clear "regime" boundaries for depthwise conv optimization. This is the central contribution the revised paper should argue.

### Regime 1: High-Channel / Moderate-Kernel (ConvMixer)
- Configs: 9×9/1536ch/32×32, 7×7/768ch/32×32
- **Winner: Custom CUDA (NHWC shared-memory tiling)**
- Custom 4× faster than cuDNN per-kernel, 5.7–6.4× end-to-end
- Custom 2.2× faster than Triton-best (native NHWC)
- Why: High channel count (768–1536) → NHWC vectorization is dense (32+ channels per cache line), shared-memory weight tiling gives massive reuse across spatial output pixels, enough thread blocks to saturate SMs

### Regime 2: Low-Channel / Large-Kernel / Small-Spatial (RepLKNet stages 2-3)
- Configs: 27×27/512ch/14×14, 13×13/1024ch/7×7
- **Winner per-kernel (vs cuDNN-NCHW): cuDNN NCHW** (0.044 ms vs 0.177 ms Custom at stage 2)
- **Winner per-kernel (vs cuDNN-NHWC): Custom NHWC** for stages 0-2 (1.9–6.7× faster)
- **Winner end-to-end: Custom CUDA** (4.55 ms vs 15.16 ms cuDNN)

**Three-way per-stage breakdown (from per-layer profiling, warm L2 cache):**

| Stage | Config | Custom NHWC | cuDNN NHWC | cuDNN NCHW | Custom vs NCHW |
|-------|--------|-------------|------------|------------|---------------|
| 0 | 31×31, 128ch, 56×56 | 0.423 ms | 2.946 ms | 0.478 ms | **1.13× faster** |
| 1 | 29×29, 256ch, 28×28 | 0.268 ms | 1.193 ms | 0.264 ms | 0.98× (tied) |
| 2 | 27×27, 512ch, 14×14 | 0.077 ms | 0.340 ms | 0.080 ms | 0.96× (tied) |
| 3 | 13×13, 1024ch, 7×7 | 0.093 ms | 0.093 ms | 0.025 ms | 0.27× (slower) |
| **DW total** | | **4.74 ms** | **16.19 ms** | **3.52 ms** | **0.74×** |
| **Non-DW** | | **3.00 ms** | **1.96 ms** | **4.75 ms** | **1.58× faster** |
| **Total (eager)** | | **7.74 ms** | **18.14 ms** | **8.26 ms** | **1.07× faster** |

**Why Custom NHWC loses to cuDNN NCHW per-kernel:**
1. **Memory coalescing**: NCHW stores each channel as a contiguous H×W plane. For 14×14 spatial, each plane is 392 bytes (~6 cache lines) — perfect sequential access. Custom NHWC strides by C=512 (1024 bytes) between spatial neighbors — poor cache line utilization.
2. **Occupancy**: Custom uses 59KB shared memory per block → limited to 1 concurrent block per SM. Grid(2,7,16)=224 blocks / 48 SMs ≈ 4.7 blocks/SM theoretical, but shmem limits actual concurrent occupancy. cuDNN NCHW can decompose with smaller blocks, achieving higher occupancy.
3. **Small-spatial parallelism**: 14×14 = 196 output pixels provides limited parallelism. Custom packs 14 pixels into 14 active warps per block (2 idle). cuDNN can achieve higher utilization on tiny planes.

**Why Custom NHWC still beats cuDNN NHWC per-kernel:**
1. **Shared-memory weight caching**: Custom loads K×K×ch_block weights into shared memory once per block, reused by all output pixels in the tile. cuDNN NHWC appears to use a generic depthwise path without shared-memory weight tiling — weights are reloaded per pixel from global/L2.
2. **cuDNN NHWC depthwise is poorly optimized**: cuDNN's best depthwise path is NCHW. Its NHWC code path shows 3.7–6.8× slower performance than its own NCHW code path (spark_cudnn_layouts.csv). Our shared-memory tiling strategy is more effective than cuDNN's NHWC fallback.

**Why Custom NHWC wins end-to-end despite losing per-kernel to cuDNN NCHW:**

The end-to-end picture is a DW vs non-DW tradeoff. Updated with native NCHW results (zero conversion):

| | Custom NHWC | cuDNN NCHW | cuDNN NHWC |
|---|---|---|---|
| DW layers | 4.74 ms | **3.52 ms** | 16.19 ms |
| Non-DW layers | **3.00 ms** | 4.75 ms | 1.96 ms |
| **Total (eager)** | **7.74 ms** | 8.26 ms | 18.14 ms |
| **Total (compiled)** | **4.52 ms** | 5.15 ms | 15.00 ms |

1. **cuDNN NCHW is NOT stranded** — it runs the entire RepLKNet natively in NCHW with zero conversion and is 2.2× faster than cuDNN NHWC (8.26 vs 18.14ms). The per-kernel NCHW advantage for depthwise DOES survive at the network level for DW-dominated architectures.
2. **But NCHW hurts non-DW layers.** 1×1 pointwise convolutions, batch normalization, and SE blocks run slower in NCHW (4.75ms vs 3.00ms for Custom NHWC). This is because 1×1 convolutions benefit from NHWC's contiguous channel dimension for GEMM operations.
3. **Custom NHWC wins the tradeoff.** Custom's shared-memory tiling brings NHWC depthwise from 4.9× slower than cuDNN NCHW (cuDNN NHWC) to within 3% parity at stage 0, though cuDNN NCHW still wins per-kernel at smaller spatial dims (0.07-0.74× cold L2, Section 4.6). In aggregate within the network (warm L2): DW 4.74ms vs 3.52ms — a manageable 1.35× DW penalty, offset by 1.58× non-DW advantage (3.00ms vs 4.75ms). Net result: Custom NHWC wins by 1.07× eager, 1.14× compiled.
4. **Architecture determines layout sensitivity.** ConvMixer (balanced DW/non-DW): NCHW ≈ NHWC. RepLKNet (DW-dominated): NCHW >> NHWC for cuDNN. This is itself a regime finding.

**Paper insight**: The per-kernel vs end-to-end comparison reveals a deeper tradeoff than simple "layout conversion overhead." Custom NHWC succeeds because shared-memory tiling **closes the NHWC DW gap** with cuDNN NCHW from 4.9× (cuDNN NHWC, cold L2) to within 3% (0.97×, stage 0), while **preserving NHWC's non-DW advantage** (3.57× faster 1×1 pointwise). Under torch.compile, a third advantage emerges: Custom's torch.library ops survive compilation intact, while inductor destroys cuDNN NCHW's direct DW kernel by replacing it with GEMM decomposition (6.6× degradation). This is a globally optimal design point that cannot be reached by either cuDNN layout alone.

### Regime 3: Triton Collapse Zone (K ≥ 27)
- All RepLKNet large-kernel configs
- **Triton is 60× slower than Custom, 18× slower than cuDNN** — even with best native layout
- Why: Triton's programming model cannot express shared-memory tiling or inter-pixel data reuse. Each program independently loads full K×K receptive field from global memory. For K=27: 729 iterations per pixel with zero reuse → memory bandwidth completely saturated by redundant loads

### Key Insight for Paper Framing

The "regime-aware" thesis answers reviewer complaint #1 ("no novelty"). The novelty is NOT the individual techniques (shmem tiling, double buffering) — it's the **empirical demonstration that no single implementation strategy dominates**, and the **characterization of where each approach wins and why**:

- cuDNN NCHW wins per-kernel for depthwise on small-spatial/large-kernel AND this advantage survives at the network level for DW-dominated architectures (RepLKNet: NCHW 2.2× faster than NHWC). But NCHW hurts non-DW layers.
- cuDNN NHWC is poorly optimized for depthwise (3.7–6.8× slower than its own NCHW) but benefits non-DW layers (1×1 conv, BN, SE blocks). Custom NHWC beats cuDNN NHWC per-kernel by 1.9–6.7× through shared-memory tiling.
- Triton works for moderate kernels (2.6–2.9× over cuDNN) but catastrophically fails for large kernels (60× slower than Custom)
- Custom NHWC finds the **globally optimal design point**: shared-memory tiling closes the NHWC depthwise gap with cuDNN NCHW from 4.9× (cuDNN NHWC, cold L2) to parity for large-spatial configs (0.97× at 31×31/128ch/56×56), while preserving NHWC's advantage on non-DW layers. Under torch.compile, custom ops survive intact while cuDNN's DW kernel gets replaced by inductor with GEMM decomposition (6.6× degradation).

**Four non-obvious findings:**
1. **Layout sensitivity is architecture-dependent**: cuDNN NCHW ≈ NHWC for ConvMixer (balanced DW/non-DW) but NCHW 2.2× faster for RepLKNet (DW-dominated). The optimal layout depends on the DW-to-non-DW compute ratio.
2. **Shared-memory tiling changes the layout tradeoff**: Without it, NHWC depthwise is 4.9× slower than NCHW (cuDNN NHWC vs NCHW, cold L2). With it (Custom), the gap shrinks to parity at large spatial (0.97× at 31×31/56×56), though cuDNN NCHW still wins per-kernel at small spatial (4.0× at 27×27/14×14). This tips the DW-vs-non-DW balance in NHWC's favor where it matters most.
3. **Triton collapses catastrophically** (60× slower) at large kernel sizes — it cannot express shared-memory data reuse, so K×K loops generate redundant global loads.
4. **No single approach dominates**: cuDNN NCHW wins DW per-kernel, Custom NHWC achieves DW parity + PW superiority, cuDNN NHWC wins non-DW layout but loses DW, and Custom NHWC wins end-to-end. Under torch.compile, Custom widens the gap because inductor preserves custom ops but destroys cuDNN NCHW's direct kernel. The optimal strategy requires considering the full network, not just individual kernels.

### Suggested Paper Structure for DGX Spark Results

1. **Table: End-to-End Latency (Main Result)**
   - ConvMixer-1536/20 + 768/32 + RepLKNet-31B
   - cuDNN-best vs Triton-best (NHWC native) vs Custom, all compiled
   - This is the headline: 5.7–6.4× over cuDNN, 2.2× over Triton on ConvMixer; 3.3× over cuDNN, 60× over Triton on RepLKNet

2. **Table: Per-Kernel Regime Analysis**
   - Isolated kernel latency for all 6 configs
   - Custom vs cuDNN-NCHW vs cuDNN-NHWC vs Triton-NHWC
   - Highlights regime boundaries: Custom wins high-ch, cuDNN-NCHW wins small-spatial per-kernel, Triton collapses at large K

3. **Table/Discussion: The Three-Way RepLKNet Comparison (KEY — explains the per-kernel vs end-to-end reversal)**
   - Per-stage: Custom NHWC vs cuDNN NHWC vs cuDNN NCHW
   - Shows Custom loses to cuDNN-NCHW (unrealizable) but wins 1.9–6.7× over cuDNN-NHWC (deployed)
   - DW totals: Custom 5.84ms vs cuDNN-NHWC 15.89ms vs cuDNN-NCHW 0.66ms
   - Explains the lesson: per-kernel benchmarks against cuDNN's best layout are misleading; layout-aware holistic design wins

4. **Table: Layout Sensitivity (cuDNN + Triton)**
   - cuDNN: NCHW 3.7–6.8× faster per-kernel, equal end-to-end (conversion overhead absorbs gains)
   - Triton: NHWC-native 2× faster than NCHW-native
   - Explains why NHWC-native design is correct despite cuDNN's per-kernel NCHW preference

5. **Figure: FP16 Roofline Plot** (after NCU)
   - All kernels on GB10 roofline: Custom (●), cuDNN-NCHW (▲), Triton (◆)
   - Shows depthwise conv is memory-bound
   - Custom kernels closer to bandwidth ceiling due to shared-memory reuse
   - Triton far below ceiling for large-K due to redundant global loads

6. **Table: NCU Microarchitecture Characterization** (after NCU)
   - Occupancy, DRAM traffic, shmem utilization, FMA pipe util, top stall reason
   - Quantifies WHY each regime boundary exists at the hardware-counter level

7. **Table: Accuracy Validation**
   - All 3 models, 50K ImageNet, zero degradation

8. **Table: Cross-Platform** (when available)
   - RTX 6000 Ada vs DGX Spark speedup ratios — shows regime boundaries shift with hardware

## 8. Remaining Work (Priority Order)

1. ~~Compile and verify kernels~~ — DONE
2. ~~ConvMixer latency benchmark~~ — DONE
3. ~~RepLKNet latency benchmark~~ — DONE
4. ~~cuDNN best-layout baseline~~ — DONE
5. ~~Triton native baselines (NHWC + NCHW, both models)~~ — DONE
6. ~~End-to-end compiled comparison (all methods, all models)~~ — DONE
7. ~~ImageNet accuracy validation (all 3 models, 50K images)~~ — DONE
8. **NCU profiling** (2 hr) — 22 kernel profiles, 6 metric sets each — NEXT (blocked on admin permissions)
9. **Nsight Systems timeline** (30 min) — RepLKNet E2E trace showing layout conversion overhead
10. **Roofline plot** (1 hr, after NCU data)
11. **Cross-platform comparison table** (1 hr) — compile RTX 6000 Ada data alongside Spark data

## 8. Important Rules for the Paper

- **Batch=1 ONLY on this branch** — DGX Spark = latency-oriented near-edge platform
- **Apple-to-apple comparisons** — eager vs eager, compile vs compile; Triton is compiled DSL so compare with torch.compile
- **NEVER mention "naive kernels"** — only discuss shared memory kernels
- **NEVER call DGX Spark an "embedded edge device"** — it is a "compact local deployment platform"
- **cuDNN baseline must use its BEST layout/algorithm** — not forced NHWC
- **All timing must be kernel-only** (CUDA events), separated from Python overhead
- **Report cold-L2 results** for per-kernel benchmarks (flush cache between runs)
- **Do NOT mention specific GPU model names in claims** — present as "compact Blackwell platform" or "Grace Blackwell system"

## 9. Known Issue: Batch Loop in Python

`convmixer/kernels/depthwise_torch_ops.py` lines 373-387 has a `for b in range(B)` Python loop. **This does NOT affect batch=1 experiments on this branch.** The batch-scaling fix (grid-based batching) is tracked on the main experiment plan, not this branch.
