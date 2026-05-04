# DGX Spark — Remaining Experiment Plan for Paper Revision

**Date**: 2026-05-04 (updated 2026-05-04)
**Branch**: SPARK
**Context**: `SPARK_context.md` — read it first for full background.

## ⚠ Critical Fixes Already Applied to This Branch

The following bugs were discovered during Ada experiments (2026-05-04) and have been fixed in this commit. **Do NOT revert these changes.** Clear any stale JIT cache before running experiments:

```bash
rm -rf ~/.cache/torch_extensions/*/depthwise_convmixer
```

### Fix 1: CUDA Stream — `depthwise_integration.cu`

**Bug**: Custom kernel launches used the default CUDA stream (stream 0). CUDA graph capture (`reduce-overhead` mode) records only the capture stream — kernels on stream 0 execute during capture but are absent on replay, producing NaN output.

**Symptom**: `torch.compile(model, mode='reduce-overhead')` gives correct output on first call, then NaN on all subsequent calls. `mode='default'` works fine (no CUDA graphs).

**Fix applied**: Added `c10::cuda::getCurrentCUDAStream()` in each torch wrapper function, passed stream as 4th `<<<grid, block, shmem, stream>>>` parameter. RepLKNet kernels already had this fix.

### Fix 2: Layout Conversion — `depthwise_torch_ops.py`

**Bug**: `CustomDepthwiseConv2d.forward()` called `x.permute(0,2,3,1).contiguous()` every forward pass. This (a) allocates new memory (incompatible with CUDA graph replay) and (b) is unnecessary — channels_last tensors already have NHWC physical layout, so `data_ptr()` gives NHWC memory directly.

**Fix applied**: Removed all permute/contiguous. Impl functions now use `torch.empty_like(input)` and pass channels_last tensors directly to C++. Forward method is single-line dispatch per op (like RepLKNet).

### Fix 3: Timing Methodology — READ CAREFULLY

**Bug**: `benchmark_compiled_comparison.py` uses bulk CUDA events (one event pair around N iterations). `convmixer/benchmark.py` uses `perf_counter` bulk timing. Both give **wrong results** for `reduce-overhead` mode.

**Root cause**: With `reduce-overhead`, CUDA graph replay returns to the CPU near-instantly. Bulk timing measures amortized throughput, not single-call latency. Error is ~1-2% for `default` mode but **up to 4×** for `reduce-overhead`.

**Correct method — use CUDA events per iteration**:
```python
times = []
for _ in range(num_runs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    model(x)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))
times.sort()
n = len(times)
latency = sum(times[n//10:n*9//10]) / len(times[n//10:n*9//10])  # trimmed mean
```

**Action**: Before running any E2E benchmark, update the benchmark function to use per-iteration CUDA events. The existing per-kernel benchmarks (direct FP16 tensors, no compile) are unaffected.

### Fix 4: Add `reduce-overhead` Mode

**Missing**: Current scripts only test `mode='default'`. The `reduce-overhead` mode (CUDA graphs) gives significant additional speedup on Ada — ConvMixer-768/32 goes from 1.26ms (default) to 0.87ms (r-o), a 1.45× improvement.

**Action**: Add `reduce-overhead` as a configuration in all E2E benchmarks. It requires Fixes 1-2 to work correctly. Note that first compilation with r-o can take 10-20 minutes (cached afterward in `/tmp/torchinductor_*/`).

### Lesson: cuDNN Best Config is NOT Always NHWC+Compile

On Ada, cuDNN's best E2E configuration for K≥9 is **NCHW eager** (not compiled, not NHWC):

| Model | cuDNN NHWC eager | cuDNN NCHW eager | cuDNN NCHW +compile |
|-------|-----------------|-----------------|---------------------|
| ConvMixer-1536/20 | 8.84 ms | **3.56 ms** | 8.51 ms (2.4× worse!) |
| RepLKNet-31B | 7.63 ms | **5.09 ms** | 6.85 ms (1.35× worse) |

`torch.compile` **degrades** cuDNN NCHW depthwise because inductor decomposes it into hundreds of small `convolve_common_engine` kernel launches. This was verified on both Ada and Spark.

**Action**: E2E layout comparison must test NCHW vs NHWC in both eager and compiled modes. Report the best cuDNN result across all 4 combinations as the baseline.

---

## 0. Status Assessment

### What Is Already Done and Reusable

All experiments from `SPARK_context.md` Sections 4.1–4.10 are complete and most data is reusable:

| Experiment | Script | FP16 Method | Reusable? |
|---|---|---|---|
| Per-kernel benchmarks (all 6 configs) | `benchmark_cudnn_layouts.py`, `benchmark_triton.py` | Direct FP16 tensors (no autocast/half distinction) | **Yes** — isolated kernel, method-independent |
| cuDNN layout study (NCHW vs NHWC) | `benchmark_cudnn_layouts.py` | Direct FP16 | **Yes** |
| Triton baselines (all 6 configs) | `benchmark_triton.py` | Direct FP16 | **Yes** |
| Compile-survival / NSYS analysis | `test_compile_fusion.py` | model.half() | **Yes** |
| NCU profiling (all profiles) | `profile_ncu.py` | Direct FP16 | **Yes** |
| Roofline plot | `plot_roofline.py` | From NCU data | **Yes** |
| Accuracy validation (3 models, 50K) | Various | N/A | **Yes** |
| RepLKNet E2E (eager + compile + per-layer) | `replknet/benchmark.py` | **model.half()** (line 264: `.to(device).half()`) | **Yes** |
| ConvMixer compiled E2E (1536/20) | `benchmark_compiled_comparison.py` | **model.half()** (lines 331–375: `.half()`) | **Yes** |
| ConvMixer compiled E2E (768/32) | `benchmark_compiled_comparison.py` | **model.half()** | **Needs verification** — check if 768/32 was run |

### What Needs Re-Doing or New Work

| Experiment | Problem | Action |
|---|---|---|
| ConvMixer EAGER benchmarks (1536/20, 768/32) | `convmixer/benchmark.py` uses autocast + perf_counter bulk timing | **Port `--fp16_method` flag + per-iteration CUDA events**, re-run with `model_half` |
| ConvMixer 768/32 compiled E2E | May not have been run; existing script uses bulk timing | **Re-run with per-iteration CUDA events** |
| **reduce-overhead E2E (all 3 models)** | **Never tested** — Fixes 1-2 required | **New**: add `reduce-overhead` mode to all E2E scripts |
| **E2E layout comparison (NCHW vs NHWC)** | cuDNN best is NCHW eager for K≥9 (see Lesson above) | **New**: test all 4 cuDNN combos (2 layouts × eager/compile) |
| Baseline models (MobileNetV3, EfficientNet, ResNet) | **Never measured on Spark** | **New**: add to convmixer/benchmark.py or standalone script |
| Tile sensitivity sweep | **Does not exist on SPARK** | **New**: copy `benchmark_sensitivity.py` from main, run on Spark |
| Cross-platform comparison table | Needs consistent methodology | **New**: compile Ada + Spark numbers |

---

## 1. Port `--fp16_method` to `convmixer/benchmark.py`

The main branch (Ada) has an updated `convmixer/benchmark.py` with `--fp16_method`, `--batch`, and `--real_data` flags. The SPARK branch version lacks these.

**Option A (recommended)**: Cherry-pick the benchmark.py changes from main:
```bash
# On the Spark machine:
git fetch origin
git cherry-pick <commit-hash-of-benchmark-changes>  # Check git log main for the commit
```

**Option B**: Manually apply these changes to `convmixer/benchmark.py`:

1. Add `--fp16_method` argument (choices: `autocast`, `model_half`, default: `model_half`)
2. In `benchmark_one()`: if `model_half`, call `x = x.half()` before benchmarking; no autocast wrapper
3. In `build_convmixer_configs()`: if `model_half`, call `model.to(dtype=torch.float16, memory_format=torch.channels_last)`
4. In `build_baseline_configs()`: respect `fp16_method` (don't hardcode `.half()` when `autocast` is requested)

The key code pattern (from Ada's `convmixer/benchmark.py`):
```python
def benchmark_one(model, device, num_runs=500, warmup=200, fp16_method='model_half', batch=1):
    x = torch.randn(batch, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    if fp16_method == 'model_half':
        x = x.half()
    # ...
    def run_forward():
        if fp16_method == 'autocast':
            with torch.amp.autocast('cuda', dtype=torch.float16):
                return model(x)
        else:
            return model(x)
```

---

## 2. Re-Run ConvMixer Eager with model.half()

After porting `--fp16_method`:

```bash
# ConvMixer-1536/20 — all 4 configs (CuDNN eager/compile, Custom eager/compile)
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 500 \
    --device cuda:0 --fp16_method model_half \
    2>&1 | tee results/spark_1536_20_model_half.log

# ConvMixer-768/32 — all 4 configs
python convmixer/benchmark.py --model 768_32 --skip_accuracy --num_runs 500 \
    --device cuda:0 --fp16_method model_half \
    2>&1 | tee results/spark_768_32_model_half.log
```

**Expected finding**: The CuDNN eager numbers will decrease (autocast overhead removed from cuDNN side). Custom eager will decrease significantly (autocast overhead on 40 pointwise layers removed). Compiled numbers should barely change (compile already fuses the pointwise casts).

Record: CuDNN eager, CuDNN+compile, Custom eager, Custom+compile latencies for both models.

---

## 3. Verify ConvMixer-768/32 Compiled (benchmark_compiled_comparison.py)

Check whether `results/spark_convmixer_768_32.txt` was generated by `benchmark_compiled_comparison.py` (model.half()) or `convmixer/benchmark.py` (autocast).

```bash
head -20 results/spark_convmixer_768_32.txt
# Look for "Compiled End-to-End Comparison" → benchmark_compiled_comparison.py (model.half())
# Or "ConvMixer-768_32 Benchmark" → convmixer/benchmark.py (autocast)
```

If it's from `convmixer/benchmark.py`, re-run with `benchmark_compiled_comparison.py`:
```bash
python benchmark_compiled_comparison.py --model 768_32 --device cuda:0 --num_runs 500 \
    --data_dir ./data/imagenet \
    2>&1 | tee results/spark_compiled_768_32_model_half.log
```

---

## 4. Baseline Models on Spark

Baselines have never been measured on Spark. These are needed for the cross-platform comparison table.

After porting `--fp16_method` to `convmixer/benchmark.py`:

```bash
# Baselines with model.half() (primary)
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 500 \
    --device cuda:0 --fp16_method model_half --baselines \
    2>&1 | tee results/spark_baselines_model_half.log
```

If `build_baseline_configs()` doesn't exist in the SPARK version, add it (copy from main's `convmixer/benchmark.py`). It loads:
- MobileNetV3-Large (`torchvision.models.mobilenet_v3_large`)
- EfficientNet-B0 (`torchvision.models.efficientnet_b0`)
- ResNet-152 (`torchvision.models.resnet152`)

Each loaded with `torch.compile(mode='default')`, `channels_last`, and `model.half()`.

---

## 5. Tile Sensitivity Sweep

Copy `benchmark_sensitivity.py` from main branch:

```bash
git show origin/main:benchmark_sensitivity.py > benchmark_sensitivity.py
```

Then run:
```bash
python benchmark_sensitivity.py --device cuda:0 --num_runs 500 --warmup 50 \
    --M_values 4 8 16 --Cb_values 32 64 128 \
    --output results/spark_sensitivity_31x31.csv \
    2>&1 | tee results/spark_sensitivity_31x31.log
```

**Expected findings** (key cross-platform comparison):
- Ada (142 SMs): M=4 fastest (784 blocks, 5.5/SM)
- Spark (48 SMs): M=8 may be faster (196 blocks, 4.1/SM — already adequate saturation)
- The Pareto-optimal tile size shifting between platforms is a regime-aware finding
- Cb=64/128 will also be skipped on Spark (SM 12.1 has ~99KB shmem limit, same as Ada)

---

## 6. Cross-Platform Comparison Table

Once Steps 2–5 are complete, build the comparison table. This is the core R3-response data.

### 6.1 E2E Model Comparison (model.half(), torch.compile)

| Model | Config | Ada (ms) | Spark (ms) | Spark/Ada ratio | Notes |
|---|---|---|---|---|---|
| ConvMixer-1536/20 | CuDNN+compile | 7.80 | ? | ? | |
| | Custom+compile | 1.59 | 4.14* | 2.60× | *verify: already model.half() |
| ConvMixer-768/32 | CuDNN+compile | 1.14 | ? | ? | |
| | Custom+compile | 1.14 | 2.47* | 2.17× | *may be autocast — verify |
| RepLKNet-31B | CuDNN+compile | 4.40 | 5.15* | 1.17× | *model.half() ✓ |
| | Custom+compile | 1.77 | 4.52* | 2.55× | *model.half() ✓ |

Ada/Spark latency ratio should reflect: 142 vs 48 SMs (2.96× SM ratio), 960 vs ~400 GB/s (2.4× BW ratio). Memory-bound kernels: expect ~2.4× ratio. Compute-bound non-DW: expect closer to SM ratio.

### 6.2 Per-Kernel Comparison (method-independent)

Already reusable from existing data:

| Config | Custom Ada (μs) | Custom Spark (μs) | Ratio |
|---|---|---|---|
| 9×9, 1536ch | 28.1 | 73 | 2.6× |
| 7×7, 768ch | 27.7 | 25 | 0.9× (Spark faster!) |
| 31×31, 128ch | 116 | 428 | 3.7× |
| 29×29, 256ch | 72 | 268 | 3.7× |
| 27×27, 512ch | 22 | 177 | 8.0× |

The 27×27 ratio (8.0×) is much larger than the SM ratio (2.96×) — this is the "no eligible warps" stall regime where Ada's 142 SMs can't be saturated by 112 blocks.

### 6.3 Tile Sensitivity Comparison

| M | Ada latency (ms) | Spark latency (ms) | Ada blocks/SM | Spark blocks/SM |
|---|---|---|---|---|
| 4 | 0.249 | ? | 5.5 | 16.3 |
| 8 | 0.297 | ? | 1.4 | 4.1 |
| 16 | 0.681 | ? | 0.45 | 1.3 |

Expected: M=8 is optimal on Spark (4.1 blocks/SM is adequate), while M=4 is optimal on Ada (needs 5.5 blocks/SM to saturate 142 SMs). This SM-count-dependent optimal tile is a key regime-aware finding.

---

## 7. What Can Be Skipped

These are NOT needed for the revision:

- **Batch scaling on Spark**: Intentionally omitted. Batch scaling story is on Ada (main branch). Spark is "latency-oriented B=1" per the branch scope.
- **Re-running per-kernel benchmarks**: Method-independent. Existing data is correct.
- **Re-running cuDNN layout study**: Method-independent. Existing data is correct.
- **Re-running Triton baselines**: Method-independent. Existing data is correct.
- **Re-running compile-survival test**: Already uses model.half(). Existing data is correct.
- **Re-running accuracy validation**: Unchanged by FP16 method.
- **Re-running NCU/NSYS**: Already collected. Existing data is correct.

---

## 8. Execution Order and Time Estimates

| Step | Description | Est. Time | Depends On |
|------|-------------|-----------|------------|
| 1 | Port --fp16_method to convmixer/benchmark.py | 15 min | — |
| 2 | Re-run ConvMixer 1536/20 + 768/32 eager with model.half() | 20 min | Step 1 |
| 3 | Verify/re-run 768/32 compiled comparison | 10 min | — |
| 4 | Add and run baselines (MobileNetV3, EfficientNet, ResNet) | 20 min | Step 1 |
| 5 | Copy and run benchmark_sensitivity.py | 30 min | — |
| 6 | Build cross-platform comparison table | 30 min | Steps 2–5 + Ada data |

**Total: ~2 hours** (most time is benchmark execution, not coding)

---

## 9. ConvMixer-768/32 (7×7) — Cross-Platform Regime Difference

This is worth special attention. **Updated Ada results (2026-05-04, CUDA events, reduce-overhead):**

**Per-kernel** (Ada):
- Custom 7×7: 18.6 μs vs cuDNN NHWC 28.9 μs → **Custom 1.55× faster**
- (Previous measurement of 27.7μs custom was less accurate)

**E2E** (Ada, CUDA events per-iteration):

| Configuration | Latency | vs cuDNN best |
|---|---|---|
| cuDNN NHWC eager | 2.52 ms | — |
| cuDNN NHWC +compile(r-o) | **1.14 ms** | 1.00× (best cuDNN) |
| Custom NHWC eager | 2.68 ms | 0.43× (**slower** than cuDNN) |
| Custom NHWC +compile(default) | 1.26 ms | 0.90× |
| **Custom NHWC +compile(r-o)** | **0.87 ms** | **1.31×** |

**Key insight**: Per-kernel custom is 1.55× faster, but E2E eager is 6% slower (Python dispatch overhead × 32 layers). The advantage only materializes with `reduce-overhead` (CUDA graphs eliminate all dispatch and launch overhead), giving 1.31× E2E speedup. This is a finding about **system-level integration** — per-kernel speedup ≠ E2E speedup until framework overhead is removed.

On Spark (cold L2, per-kernel):
- Custom 25μs vs cuDNN NCHW 71μs → **Custom 2.84× faster**
- cuDNN NHWC: 362μs → Custom 14.5× faster

**Cross-platform regime finding**: On Ada, cuDNN has a fast 7×7 NHWC kernel (28.9μs), so custom's per-kernel advantage is modest (1.55×). On Spark, cuDNN's 7×7 is much slower (71μs NCHW), so custom wins by 2.84×. The optimal regime boundary is hardware-dependent.

**For Spark experiments**: Make sure to test reduce-overhead mode for 768/32. If Spark shows similar eager-slower/r-o-faster behavior, it confirms the system-level finding is platform-independent. If Spark's eager is already faster (fewer layers × lower dispatch overhead per call), that's also interesting data.

---

## 10. Output Files Expected

After completing all steps:

```
results/
├── spark_1536_20_model_half.log           ⬜ (Step 2)
├── spark_768_32_model_half.log            ⬜ (Step 2)
├── spark_compiled_768_32_model_half.log   ⬜ (Step 3, if needed)
├── spark_baselines_model_half.log         ⬜ (Step 4)
├── spark_sensitivity_31x31.csv            ⬜ (Step 5)
├── spark_sensitivity_31x31.log            ⬜ (Step 5)
├── spark_cross_platform_table.md          ⬜ (Step 6)
│
│ (existing — reusable)
├── spark_hardware_info.txt                ✅
├── spark_convmixer_1536_20.txt            ✅ (autocast — keep for reference)
├── spark_convmixer_1536_20_v2.txt         ✅ (model.half() compiled)
├── spark_convmixer_768_32.txt             ✅ (autocast — keep for reference)
├── spark_replknet.txt                     ✅ (model.half())
├── spark_cudnn_layouts.csv                ✅ (method-independent)
├── spark_triton_results.csv               ✅ (method-independent)
├── spark_kernel_comparison.csv            ✅ (method-independent)
├── spark_ncu_metrics.csv                  ✅
├── spark_ncu_tables.md                    ✅
├── spark_roofline_data.csv                ✅
├── spark_roofline_plot.png                ✅
└── ncu/                                   ✅
```
