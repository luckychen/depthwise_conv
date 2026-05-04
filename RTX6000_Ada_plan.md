# RTX 6000 Ada Experiment Plan — Cross-Platform Pair for Revision

This file is the executable instruction sheet for the RTX 6000 Ada workstation. It mirrors the experiments already completed on DGX Spark (`SPARK` branch, `SPARK_context.md`) so that the revised paper can present a two-platform regime study.

**Read this entire file before doing anything.** Follow the numbered action items in Section 4.9 exactly. Do not improvise or skip steps.

## 0. Why Ada Matters for the Revision

Reviewer R3 said "single GPU" was a reject-grade flaw. The revision needs a side-by-side cross-platform table. Ada (SM 8.9, 142 SMs, 96 MB L2, 960 GB/s, 91.1 FP16 TFLOPS) and Spark (SM 12.1, 48 SMs, unified Grace-Blackwell memory, ~209 FP16 TFLOPS) are deliberately different — different SM count, different L2 size, different memory subsystem. **If regime boundaries shift between the two platforms, that shift is itself a finding** and confirms the regime-aware thesis.

**Status (2026-05-03)**:
- Steps 4.1–4.9: ✅ Complete
- Step 4.8 (tile sensitivity): ⬜ Not started — **R1 specifically requested this, HIGH priority**
- Roofline plot: ⬜ Blocked on matplotlib

**Compile-survival confirmed cross-platform**: `torch.compile` decomposes cuDNN NCHW into 768+ GEMM kernels. Ada shows **9.2× degradation** (167.7 → 1543.2 μs), Spark showed 6.6×. Platform-independent finding confirmed.

## 1. Hardware Specs (Reference)

| Property | Ada (this machine) | Spark |
|---|---|---|
| GPU | NVIDIA RTX 6000 Ada | DGX Spark (GB10) |
| Architecture | Ada Lovelace (SM 8.9) | Blackwell (SM 12.1) |
| SM Count | 142 | 48 |
| L2 Cache | 96 MB | ~22 MB equivalent |
| Memory | 48 GB GDDR6 | 128 GB unified |
| Memory Bandwidth | 960 GB/s | ~400 GB/s estimated |
| FP16 Peak | 91.1 TFLOPS | ~209 TFLOPS |

**Ada has 3x the SMs and 4.4x the L2 of Spark, but ~half the FP16 peak.**

## 2. Files from SPARK Branch — ✅ Already Copied

All 8 scripts are present in the working directory:
```
benchmark_cudnn_layouts.py, benchmark_triton.py, benchmark_compiled_comparison.py,
profile_ncu.py, run_ncu_profiles.sh, analyze_ncu.py, plot_roofline.py, test_compile_fusion.py
```

## 3. Environment — ✅ Confirmed Working

```bash
source /home/chenchon/miniconda3/etc/profile.d/conda.sh && conda activate cuda
export CUDA_VISIBLE_DEVICES=1   # physical GPU 1; do NOT add to ~/.bashrc
cd /home/chenchon/chConv/paper/latex/repo
```

Stack confirmed: Python 3.10, PyTorch 2.9.1+cu128, CUDA 12.8 (nvcc 12.9), cuDNN 9.10.2.

### FP16 Methodology — RESOLVED for ConvMixer-1536/20

Two methods exist. **Use `model.half()` as the primary methodology** for all revision numbers going forward. Autocast numbers are kept as a secondary column for continuity with the original submission.

| Method | How | When to use |
|--------|-----|-------------|
| **model.half()** (primary) | `model = model.half()` then `model(x.half())` | All revision numbers. Deployment-representative. No per-forward overhead. |
| **autocast** (secondary) | FP32 model, `torch.amp.autocast` per forward | Paper continuity column only. Has per-forward FP32->FP16 cast overhead on pointwise layers. |

**Root cause of difference**: `replace_depthwise_convs` stores custom depthwise weights as FP16 already. The autocast overhead falls entirely on the 40 pointwise Conv2d layers (still FP32). `torch.compile` fuses these and eliminates the overhead, which is why the compiled gap is small (1.62 -> 1.47 ms) while the eager gap is large (3.60 -> 1.90 ms).

**ConvMixer-1536/20 B=1 (both methods, verified 2026-05-03):**

| Configuration | autocast latency | model.half() latency |
|---------------|-----------------|---------------------|
| CuDNN eager | 9.07 ms | 7.40 ms |
| CuDNN + compile | 7.55 ms | 7.23 ms |
| Custom eager | 3.60 ms | 1.90 ms |
| Custom + compile | 1.62 ms | 1.47 ms |

**Benchmark flags** (updated `convmixer/benchmark.py`):
```bash
# model.half() (primary — use this for all new measurements)
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 500 --device cuda:0 --fp16_method model_half

# autocast (secondary — paper continuity)
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 500 --device cuda:0 --fp16_method autocast

# with real ImageNet data, batch=N, GPU pre-loaded (no CPU->GPU transfer in timing)
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 --device cuda:0 \
    --fp16_method model_half --batch 4 --real_data --data_dir /home/chenchon/chConv/data/imagenet
```

## 4. Experiments

### 4.1 Verify Custom Kernels — ✅ Complete

**Results**: Custom+compile latencies match paper targets.
- ConvMixer-1536/20: Custom+compile 1.62 ms (autocast), 1.47 ms (model.half())
- ConvMixer-768/32: pass
- RepLKNet-31B: pass

Output files: `results/ada_convmixer_1536_20_smoke.txt`, `ada_convmixer_768_32_smoke.txt`, `ada_replknet_smoke.txt`

### 4.2 cuDNN Best-Layout per Kernel — ✅ Complete

| Config | NCHW (ms) | NHWC (ms) | Best |
|--------|-----------|-----------|------|
| 9x9/1536/32x32 | 0.090 | 0.289 | **NCHW** (3.19x) |
| 7x7/768/32x32 | 0.027 | 0.018 | **NHWC** (1.47x) |
| 31x31/128/56x56 | 0.150 | 0.634 | **NCHW** (4.23x) |
| 29x29/256/28x28 | 0.077 | 0.267 | **NCHW** (3.49x) |
| 27x27/512/14x14 | 0.015 | 0.093 | **NCHW** (6.06x) |

**Finding**: NCHW dominates for all K>7. Custom NHWC approach is fighting cuDNN's best layout.

Output file: `results/ada_cudnn_layouts.csv`

### 4.3 Triton Baseline per Kernel — ✅ Complete

| Config | Triton (ms) | Custom (ms) | Custom/Triton |
|--------|------------|-------------|---------------|
| 9x9/1536/32x32 | 0.127 | 0.028 | 4.5x |
| 7x7/768/32x32 | 0.043 | 0.028 | 1.5x |
| 31x31/128/56x56 | 0.286 | 0.116 | 2.5x |
| 27x27/512/14x14 | 0.038 | 0.022 | 1.7x |
| 13x13/1024/7x7 | 0.022 | 0.073 | 0.30x (native cuDNN used) |

Output file: `results/ada_triton_results.csv`

### 4.4 End-to-End Compiled Comparison — ✅ Complete (autocast only)

Output files: `results/ada_compiled_1536_20.log`, `ada_compiled_768_32.log`, `ada_compiled_replknet.log`

**Note**: These were all run with autocast. The model.half() equivalents for 768/32 and RepLKNet are collected in Step 4.9.

### 4.5 NCU Profiling — ✅ Complete (15/17 profiles)

See results.md Section 9 for full metrics. Key findings:
- Custom 9x9/1536: 3628 GFLOP/s at 55% occupancy (3.4x higher than cuDNN's 1066 GFLOP/s at 62% occupancy)
- Custom 27x27/512: 75% "no eligible warps" stalls — 112 blocks for 142 SMs, spatial parallelism regime boundary

**Roofline plot**: Blocked on matplotlib. Run after `pip install matplotlib`:
```bash
python plot_roofline.py --ncu_csv results/ada_ncu_metrics.csv \
    --output results/ada_roofline_plot.png \
    --fp16_peak_tflops 91.1 --bandwidth_gbps 960
```

Output files: `results/ada_ncu_metrics.csv`, `ada_ncu_tables.md`, `ncu/*.csv`

### 4.6 NSYS Compile-Survival Verification — ✅ Complete

| Method | Eager (us) | Compiled (us) | Result |
|--------|-----------|---------------|--------|
| Custom NHWC | 163.7 | 126.0 | **1.30x faster** |
| cuDNN NCHW | 167.7 | **1543.2** | **9.2x degradation** |

**Compile-survival is platform-independent.** Lead finding for revision.

Output file: `results/ada_nsys_compile_fusion.log`

### 4.7 Batch Sweep — ✅ Complete (ConvMixer-1536/20 only)

BATCH_GROUP=2 kernel implemented. Correctness verified B=1-16.

**Important caveat discovered**: BATCH_GROUP=2 processes batches sequentially within each group (not in parallel). It reduces kernel launches (4->2 for B=4) and amortizes weight loads, but does NOT increase compute parallelism. Throughput gain is therefore modest.

**Results (model.half(), real ImageNet, GPU pre-loaded):**

| B | Custom eager (ms) | Custom eager (img/s) | Custom+compile (ms) | Custom+compile (img/s) | CuDNN eager (ms) | CuDNN eager (img/s) |
|---|------------------|---------------------|--------------------|-----------------------|-----------------|---------------------|
| 1 | 1.90 | 526 | **1.47** | **682** | 7.40 | 135 |
| 4 | 6.80 | **588** | 6.62 | 604 | 31.95 | 125 |

**Finding**: B=4 gives only 12% throughput gain over B=1 eager (526->588 img/s). Compiled B=1 (682 img/s) remains peak. CuDNN throughput declines with batch.

**Previous claim of 1315 img/s at B=4 was wrong** — not reproducible. Discarded.

Output files: `results/ada_1536_20_*.log`

### 4.8 Tile-Size and Channel-Blocking Sensitivity — ⬜ HIGH PRIORITY

**R1 specifically asked for this.** Missing it risks a second reject on the same point.

Sweep on 31x31/128ch/56x56 (most shared-memory-pressured config). Vary M in {4, 8, 16}, Cb in {32, 64, 96}. For each (M, Cb), recompile the kernel and report latency, occupancy, shmem, registers.

Write `benchmark_sensitivity.py` (does not exist yet). For each (M, Cb) combination:
1. Recompile kernel with new template parameters
2. Run 500 iterations with cold L2 (flush between runs)
3. Optionally run NCU `--section Occupancy` for achieved occupancy
4. Record: latency, static shmem per block, registers per thread, achieved occupancy

```bash
python benchmark_sensitivity.py --config 31x31_128_56 --device cuda:0 --output results/ada_sensitivity_31x31.csv
```

Expected: Pareto front. If (M=8, Cb=64) is optimal on Ada, that justifies the choice. If a different point is optimal, that supports the regime-aware thesis.

**Time estimate**: ~3 hours including kernel recompiles.

### 4.9 Methodology Consistency Pass — ⬜ CRITICAL, DO THIS FIRST

**Problem**: The revision uses `model.half()` as the primary methodology, but only ConvMixer-1536/20 has been measured with it. The other two models (768/32 and RepLKNet) and the baseline comparison table are still autocast-only or inconsistently mixed. The speedup ratios in the paper will change when both sides (CuDNN and Custom) use model.half(). This MUST be resolved before any paper numbers are finalized.

**Execute these steps in order. Do not skip any.**

#### Step 1: ConvMixer-768/32, model.half(), B=1 (~10 min)

```bash
source /home/chenchon/miniconda3/etc/profile.d/conda.sh && conda activate cuda
export CUDA_VISIBLE_DEVICES=1
cd /home/chenchon/chConv/paper/latex/repo

python convmixer/benchmark.py --model 768_32 --skip_accuracy --num_runs 500 --device cuda:0 \
    --fp16_method model_half 2>&1 | tee results/ada_768_32_model_half_b1.log
```

Record the 4 latencies (CuDNN eager, CuDNN+compile, Custom eager, Custom+compile).

#### Step 2: RepLKNet-31B, model.half(), B=1 (~15 min)

The RepLKNet benchmark script (`replknet/benchmark.py`) does NOT have the `--fp16_method` flag yet. You need to either:
- (a) Add the same `--fp16_method model_half` support to `replknet/benchmark.py` following the pattern in `convmixer/benchmark.py`, OR
- (b) Modify `replknet/benchmark.py` temporarily to call `model.half()` before benchmarking and remove the autocast wrapper

After adding model.half() support:
```bash
python replknet/benchmark.py --device cuda:0 --num_runs 100 --skip_verify --skip_accuracy \
    --fp16_method model_half --per_layer 2>&1 | tee results/ada_replknet_model_half_b1.log
```

Record total latency and per-stage breakdown.

#### Step 3: Baseline models with autocast, for consistency (~15 min)

The baseline models (MobileNetV3, EfficientNet, ResNet) in `convmixer/benchmark.py:build_baseline_configs()` currently use `model.half()` (lines 202, 208, 214 — `.to(device, dtype=torch.float16)`). But the ConvMixer entries in the same comparison table used autocast. This is an apples-to-oranges comparison.

**Two options** — pick ONE and apply it consistently:

**Option A (recommended)**: Make baselines also use autocast when `--fp16_method autocast`, and model.half() when `--fp16_method model_half`. This means modifying `build_baseline_configs()` to respect the `fp16_method` argument:
- When `model_half`: keep current behavior (`.to(dtype=torch.float16)`)
- When `autocast`: load as FP32, let autocast handle it

**Option B**: Just measure everything with model.half() and don't report autocast baselines.

After implementing, run:
```bash
# model.half() baselines (should match existing numbers since baselines already use .half())
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 500 --device cuda:0 \
    --fp16_method model_half --baselines 2>&1 | tee results/ada_baselines_model_half.log

# autocast baselines (NEW — baselines were never measured with autocast)
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 500 --device cuda:0 \
    --fp16_method autocast --baselines 2>&1 | tee results/ada_baselines_autocast.log
```

#### Step 4: ConvMixer-1536/20, model.half(), B=8 (~5 min)

One more batch data point to show L2 behavior at larger working sets:
```bash
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 --device cuda:0 \
    --fp16_method model_half --batch 8 --real_data --data_dir /home/chenchon/chConv/data/imagenet \
    2>&1 | tee results/ada_1536_20_real_model_half_b8.log
```

#### Step 5: Update results.md with all new data

After Steps 1-4, update `revision/results.md`:

1. **Section 3**: Add a model.half() column to all three model tables (3.1, 3.2, 3.3), or replace autocast with model.half() as primary. Note the changed speedup ratios.

2. **Section 4 (Baseline Comparison)**: Rebuild the table using consistent methodology. Every row must use the same FP16 method. Recompute Acc/ms with the new latencies.

3. **Section 5.2**: Add B=8 row.

4. Verify all speedup ratios are recomputed from the new numbers. The headline "5.37x" will likely change — report the correct number.

#### Step 6: Update context.md

Mark 4.9 complete. Update the "What Remains" section.

### Summary of what must still be done

| Step | Priority | Time | Status | Why |
|------|----------|------|--------|-----|
| **4.9** (methodology consistency) | **CRITICAL** | ~1 hour | ✅ DONE | All three models + baselines now use model.half() consistently |
| **4.8** (tile sensitivity) | **HIGH** | ~3 hours | 🔄 Running | R1 explicitly requested this; `benchmark_sensitivity.py` created and running |
| **SPARK: re-run model.half()** | **HIGH** | ~1 hour | ⬜ Needs Spark session | Cross-platform table needs same methodology on both platforms |
| Install matplotlib + roofline | Medium | 15 min | ⬜ | Visual for paper, not blocking |
| **SPARK: run benchmark_sensitivity.py** | Medium | ~1 hour | ⬜ Needs Spark session | Cross-platform tile-sensitivity comparison |
| **SPARK: NCU profiling** | Medium | ~2 hours | ⬜ Blocked (admin) | Need sudo for NCU on DGX Spark |
| 7x7/RepLKNet batch kernels | Low | 4-6 hours | ⬜ | The batch story is modest (+12%), not worth large investment |

## 5. Cross-Platform Comparison — Ada Results Filled In

| Finding | Spark behavior | Ada result | Cross-platform? |
|---|---|---|---|
| Compile-survival (cuDNN NCHW destroyed by inductor) | 768 GEMM launches, 6.6x degradation | **9.2x degradation confirmed** | Yes, platform-independent |
| Layout isolation (cuDNN NCHW > NHWC for K>7) | NCHW wins for 9x9+ | **NCHW dominates all K>7** | Yes |
| Custom wins per-kernel high-ch/moderate-K | 4x over cuDNN-best on 9x9/1536 | **3.2x per-kernel, ~5x E2E compiled (model.half())** | Yes |
| Custom loses for low-ch/large-K/small-spatial | 0.07x on 27x27/512/14x14 (Spark) | **0.68x on 27x27/512**; NCU: 75% no-eligible stalls | Yes, regime boundary confirmed |
| Triton-collapse on K>=27 | 60x slower on RepLKNet | **2.5x slower on 31x31, 1.7x on 27x27** | Partial (Ada L2 helps Triton) |
| Batch scaling | TBD | **B=1 compiled optimal (682 img/s); B=4 +12% eager only** | TBD vs Spark |

## 6. Output Files

```
results/
├── ada_hardware_info.txt                    ✅
├── ada_convmixer_1536_20_smoke.txt          ✅
├── ada_convmixer_768_32_smoke.txt           ✅
├── ada_replknet_smoke.txt                   ✅
├── ada_cudnn_layouts.csv                    ✅
├── ada_triton_results.csv                   ✅
├── ada_compiled_1536_20.log                 ✅
├── ada_compiled_768_32.log                  ✅
├── ada_compiled_replknet.log                ✅
├── ada_ncu_metrics.csv                      ✅
├── ada_ncu_tables.md                        ✅
├── ada_ncu_run.log                          ✅
├── ada_roofline_plot.png                    ⬜ (matplotlib not installed)
├── ada_nsys_compile_fusion.log              ✅
├── ada_1536_20_autocast.log                 ✅
├── ada_1536_20_model_half.log               ✅
├── ada_1536_20_autocast_b4.log              ✅
├── ada_1536_20_model_half_b4.log            ✅
├── ada_1536_20_real_model_half_b1.log       ✅
├── ada_1536_20_real_model_half_b4.log       ✅
├── ada_768_32_model_half_b1.log             ✅ (Step 4.9.1)
├── ada_replknet_model_half_b1.log           ✅ (Step 4.9.2 — RepLKNet already used model.half())
├── ada_baselines_model_half.log             ✅ (Step 4.9.3)
├── ada_baselines_autocast.log               ✅ (Step 4.9.3)
├── ada_1536_20_real_model_half_b8.log       ✅ (Step 4.9.4)
├── ada_sensitivity_31x31.csv                🔄 (Step 4.8 — in progress)
├── ada_sensitivity_31x31.log                🔄 (Step 4.8 — in progress)
└── ncu/                                     ✅ (15/17 profiles)
```

## 7. What SPARK Branch Needs (Next Spark Session)

SPARK branch experiments are complete with autocast methodology. For the cross-platform paper table, Spark results must use `model.half()` to match Ada. This requires a session on the Spark machine.

### Step 1: Merge methodology changes from main

The `--fp16_method` flag, `--batch`, `--real_data` flags, and `build_baseline_configs()` fix are on `main` but not `SPARK`. Options:
- Cherry-pick the relevant commits: `git cherry-pick <convmixer_benchmark_commit>`
- Or manually apply the same changes to SPARK's `convmixer/benchmark.py`

### Step 2: Re-run ConvMixer with model.half()

```bash
# ConvMixer-1536/20
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 500 \
    --device cuda:0 --fp16_method model_half \
    2>&1 | tee results/spark_1536_20_model_half.log

# ConvMixer-768/32
python convmixer/benchmark.py --model 768_32 --skip_accuracy --num_runs 500 \
    --device cuda:0 --fp16_method model_half \
    2>&1 | tee results/spark_768_32_model_half.log
```

RepLKNet benchmark.py on SPARK should already use model.half() (load_model() calls .half() directly — verify before running).

### Step 3: Re-run baselines with model.half()

```bash
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 500 \
    --device cuda:0 --fp16_method model_half --baselines \
    2>&1 | tee results/spark_baselines_model_half.log
```

### Step 4: Tile sensitivity sweep

Copy `benchmark_sensitivity.py` from main and run:

```bash
python benchmark_sensitivity.py --device cuda:0 --num_runs 500 --warmup 50 \
    --M_values 4 8 16 --Cb_values 32 64 128 \
    --output results/spark_sensitivity_31x31.csv \
    2>&1 | tee results/spark_sensitivity_31x31.log
```

Expected finding: Spark's smaller L2 (~22 MB vs Ada's 96 MB) should shift the Pareto-optimal (M, Cb) point — larger Cb may hurt more on Spark due to L2 saturation. This cross-platform tile sensitivity difference is itself a finding.

### Step 5: Build cross-platform comparison table

Once both platforms have model.half() numbers, compile the table. Expected ratio:
- ConvMixer-1536/20 (memory-BW bound): Ada/Spark ≈ 960/400 = 2.4× (Ada faster)
- RepLKNet (mixed DW + non-DW): ratio may differ due to non-DW compute sensitivity
- Custom+compile speedup over cuDNN: expect similar ratios — regime-aware thesis should hold on both

### Step 6: NCU profiling (if admin access available)

```bash
bash run_ncu_profiles.sh 2>&1 | tee results/spark_ncu_run.log
python analyze_ncu.py --ncu_dir results/ncu/ --output results/spark_ncu_metrics.csv
```

If admin still blocked, use Ada NCU data for the paper's profiling section and note Spark NCU was blocked.

### SPARK Output Files Needed

```
results/
├── spark_1536_20_model_half.log     ⬜ (Step 7.2)
├── spark_768_32_model_half.log      ⬜ (Step 7.2)
├── spark_baselines_model_half.log   ⬜ (Step 7.3)
├── spark_sensitivity_31x31.csv      ⬜ (Step 7.4)
├── spark_sensitivity_31x31.log      ⬜ (Step 7.4)
```

---

## 8. Known Pitfalls

1. **Stale ninja lock**: If JIT compile hangs at 0%, delete: `rm -rf ~/.cache/torch_extensions/py310_cu128/depthwise_*`
2. **torch.compile `reduce-overhead` on RepLKNet**: First compile takes 10-20 min. Cache in `/tmp/torchinductor_chenchon/`.
3. **GPU 0 occupied**: Another process holds 17 GB on GPU 0. Always `CUDA_VISIBLE_DEVICES=1`.
4. **benchmark.py throughput field**: Was reporting batches/s as img/s — fixed 2026-05-03. Now correctly reports `(num_runs * batch) / elapsed`.
5. **Data loading**: Always pre-load batches to GPU before timing. CPU->GPU transfer excluded from all benchmarks (correct: identical for all configs, overlapped in real pipelines, zero-cost on unified-memory edge devices).
6. **build_baseline_configs() fix — DONE**: Was unconditionally calling `.to(dtype=torch.float16)`. Fixed in Step 4.9.3 to respect `--fp16_method`.
7. **replknet/benchmark.py on SPARK**: SPARK branch benchmark.py differs from Ada's. RepLKNet load_model() calls .half() directly — verify before re-running.
8. **benchmark_sensitivity.py Cb=96 invalid**: 128 % 96 = 32 (doesn't divide C=128). Use {32, 64, 128} instead.
9. **benchmark_sensitivity.py correctness threshold**: 31×31 with 961 FP16 MACs gives max_diff ≈ 1.3. Threshold set to 3.0 to accommodate this.
