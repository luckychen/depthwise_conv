# DGX Spark — Remaining Experiment Plan for Paper Revision

**Date**: 2026-05-08 (updated — L2 residency reframe, batch sweep added)
**Branch**: SPARK
**Context**: `SPARK_context.md` — read first for full background.

## ⚠ Critical: Read SPARK_plan.md CAREFULLY Before Running Any Experiment

The SPARK branch has fixes applied for CUDA stream (CUDA graph compatibility) and layout conversion (remove permute/contiguous). **Do NOT revert these.** Clear stale JIT cache before running:

```bash
rm -rf ~/.cache/torch_extensions/*/depthwise_convmixer
```

Also: use **per-iteration CUDA events** (not bulk timing) for all E2E benchmarks. Bulk timing gives wrong results for `reduce-overhead` mode (up to 4× error).

---

## 0. Scientific Motivation (Why These Experiments)

The paper's central claim: **shared-memory weight tiling matters when (working set) > (L2 capacity).**

| Platform | L2 | Working set (9×9/1536ch, B=1) | Resident? |
|----------|----|-------------------------------|-----------|
| RTX 6000 Ada | **96 MB** | ~1.5 MB | Yes (L2 absorbs everything) |
| DGX Spark | **~24 MB** | ~1.5 MB | Yes at B=1, **exceeds at B=4** |

On Ada (large L2): batch scaling gives marginal gain (everything stays L2-resident).
On Spark (small L2): batch should push the working set beyond L2 → weight caching matters more → custom kernel advantage GROWS with batch.

The Spark experiments test this prediction. **This is the key new evidence distinguishing the revision from the original submission.**

---

## 0. Status Assessment

### Already Done and Reusable

All experiments from `SPARK_context.md` §§4.1–4.10 are complete. Most data is reusable:

| Experiment | FP16 Method | Reusable? |
|---|---|---|
| Per-kernel benchmarks (all 6 configs) | Direct FP16 tensors | **Yes** — method-independent |
| cuDNN layout study (NCHW vs NHWC) | Direct FP16 | **Yes** |
| Triton baselines (all 6 configs) | Direct FP16 | **Yes** |
| Compile-survival / NSYS analysis | model.half() | **Yes** |
| NCU profiling (all profiles) | Direct FP16 | **Yes** |
| Roofline plot | From NCU data | **Yes** |
| Accuracy validation (3 models, 50K) | N/A | **Yes** |
| RepLKNet E2E (eager + compile + per-layer) | **model.half()** | **Yes** |
| ConvMixer compiled E2E (1536/20) | **model.half()** | **Yes** |

### What Needs Doing

| Experiment | Problem | Priority |
|---|---|---|
| ConvMixer EAGER benchmarks (1536/20, 768/32) | Uses autocast | **CRITICAL** — port `--fp16_method`, re-run with `model_half` |
| **Batch sweep B=1,2,4 on Spark** | **Never run** — this is the L2 hypothesis test | **CRITICAL** — new addition |
| Baseline models (MobileNetV3, EffNet, ResNet) | Never measured on Spark | **CRITICAL** — needed for cross-platform table |
| Tile sensitivity sweep | Does not exist on SPARK | **HIGH** — cross-platform tile comparison |
| **L2 hit rate extraction from NCU** | **Data exists, not extracted** | **HIGH** — quantitative backbone for L2 claim |

---

## 1. Port `--fp16_method` to `convmixer/benchmark.py` (~15 min)

The main branch (Ada) has `--fp16_method`, `--batch`, and `--real_data` flags. Port them to SPARK.

**Option A (recommended)**: Cherry-pick from main:
```bash
git fetch origin
git cherry-pick <commit-hash>  # Check git log main for the benchmark changes
```

**Option B**: Manually apply:

1. Add `--fp16_method` argument (choices: `autocast`, `model_half`, default: `model_half`)
2. In `benchmark_one()`: if `model_half`, call `x = x.half()` before benchmarking; no autocast wrapper
3. In `build_convmixer_configs()`: if `model_half`, call `model.to(dtype=torch.float16, memory_format=torch.channels_last)`
4. In `build_baseline_configs()`: respect `fp16_method`

Key code pattern from Ada:
```python
def benchmark_one(model, device, num_runs=500, warmup=200, fp16_method='model_half', batch=1):
    x = torch.randn(batch, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    if fp16_method == 'model_half':
        x = x.half()
    def run_forward():
        if fp16_method == 'autocast':
            with torch.amp.autocast('cuda', dtype=torch.float16):
                return model(x)
        else:
            return model(x)
```

Also add per-iteration CUDA events timing (see `final_revision_plan.md` §8 of context.md for the pattern).

---

## 2. Re-Run ConvMixer Eager with model.half() (~20 min)

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

Record: CuDNN eager, CuDNN+compile, Custom eager, Custom+compile latencies.

**Expected**: CuDNN eager numbers decrease (autocast overhead removed). Custom eager decreases significantly (pointwise autocast overhead removed from 40 layers). Compiled numbers barely change.

---

## 3. Batch Sweep on Spark — L2 Hypothesis Test (~30 min) ⭐ NEW

**This is the most important new experiment.** It tests whether batch size pushes the working set beyond Spark's ~24MB L2, making weight caching more valuable.

**Run ConvMixer-1536/20 with real ImageNet data, model.half(), B=1,2,4:**

```bash
# B=1 (baseline — should match Step 2)
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 1 --real_data \
    --data_dir ./data/imagenet \
    2>&1 | tee results/spark_batch_b1.log

# B=2
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 2 --real_data \
    --data_dir ./data/imagenet \
    2>&1 | tee results/spark_batch_b2.log

# B=4
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 4 --real_data \
    --data_dir ./data/imagenet \
    2>&1 | tee results/spark_batch_b4.log
```

For each B, record: Custom eager (ms, img/s), Custom+compile (ms, img/s), cuDNN eager (ms, img/s).

### What to Look For

| Prediction | Ada (96MB L2) result | Expected Spark (~24MB L2) result |
|---|---|---|
| B=1 → B=4 throughput gain | +12% (526→588 img/s eager) | **Larger than 12%** if L2 hypothesis holds |
| Custom+compile at B=4 vs B=1 | B=4 is 604 vs B=1 682 img/s (worse!) | **B=4 should be BETTER than B=1** if weight caching amortization matters |
| cuDNN batch scaling | Flat/declining (125 img/s) | Should also be flat — cuDNN doesn't cache weights across batch |

### How to Interpret

- **If Spark B=4 throughput > B=1 throughput for Custom** → L2 hypothesis confirmed. Weight caching matters more when L2 is small and batch expands the working set. This is the headline finding.
- **If Spark B=4 ≈ B=1** → Bottleneck is bandwidth (not capacity). The paper adjusts: "On bandwidth-constrained platforms, shared-memory tiling helps even at B=1 because redundant DRAM loads dominate regardless of L2."
- **Either outcome is publishable.** The experiment's value is in testing the hypothesis, not confirming it.

---

## 4. Baseline Models on Spark (~20 min)

```bash
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 500 \
    --device cuda:0 --fp16_method model_half --baselines \
    2>&1 | tee results/spark_baselines_model_half.log
```

Measures: MobileNetV3-Large, EfficientNet-B0, ResNet-152. All with `torch.compile(mode='default')`, `channels_last`, `model.half()`.

---

## 5. Tile Sensitivity Sweep (~30 min)

Copy `benchmark_sensitivity.py` from main branch:
```bash
git show origin/main:benchmark_sensitivity.py > benchmark_sensitivity.py
```

Run:
```bash
python benchmark_sensitivity.py --device cuda:0 --num_runs 500 --warmup 50 \
    --M_values 4 8 16 --Cb_values 32 \
    --output results/spark_sensitivity_31x31.csv \
    2>&1 | tee results/spark_sensitivity_31x31.log
```

**Expected cross-platform finding**:

| M | Ada blocks/SM | Spark blocks/SM | Ada optimal? | Spark optimal? |
|---|--------------|-----------------|-------------|----------------|
| 4 | 5.5 | 16.3 | **Yes** (needs saturation) | Diminishing returns |
| 8 | 1.4 | 4.1 | Marginal | **Expected optimal** |
| 16 | 0.45 | 1.3 | Catastrophic | Poor |

Pareto-optimal tile size shifts with SM count → regime-aware finding.

---

## 6. Extract L2 Hit Rates from Existing Spark NCU Data (~15 min) ⭐ NEW

The Spark NCU data is already collected (`results/ncu/*.csv`, `results/spark_ncu_metrics.csv`). Extract L2-specific metrics:

```bash
# If analyze_ncu.py supports L2 extraction:
python analyze_ncu.py --ncu_dir results/ncu/ --output results/spark_ncu_l2.csv --metric l2

# Otherwise, grep from raw CSVs:
for f in results/ncu/*.csv; do
    echo "=== $(basename $f) ==="
    grep -i "l2\|lts" "$f" | head -5
done
```

Target metrics: `lts__t_sectors`, `lts__t_sectors_lookup_hit_rate`, or equivalent L2 cache counters.

### Build Cross-Platform L2 Table

| Config | Ada 96MB L2 hit (Custom) | Spark ~24MB L2 hit (Custom) | Δ |
|--------|------------------------|---------------------------|------|
| 9×9, 1536ch | 81.7% | ? | ? |
| 7×7, 768ch | 75.9% | ? | ? |
| 31×31, 128ch | 95.9% | ? | ? |
| 27×27, 512ch | 93.0% | ? | ? |

**This is the paper's central evidence table.** If Spark L2 hit rates are significantly lower than Ada's, the L2 residency principle is quantitatively validated.

---

## 7. Build Cross-Platform Comparison Table (~30 min)

After Steps 2–6, compile the master comparison table:

### E2E Model Comparison (model.half(), torch.compile reduce-overhead)

| Model | Config | Ada (ms) | Spark (ms) | Spark/Ada |
|-------|--------|---------|------------|-----------|
| ConvMixer-1536/20 | CuDNN best | 3.56 (NCHW eager) | ? | ? |
| | Custom+compile(r-o) | 1.89 | ? | ? |
| ConvMixer-768/32 | CuDNN+compile(r-o) | 1.14 | ? | ? |
| | Custom+compile(r-o) | 0.87 | ? | ? |
| RepLKNet-31B | CuDNN best | 5.09 (NCHW eager) | ? | ? |
| | Custom+compile(r-o) | 1.62 | ? | ? |

### Baseline Comparison (model.half(), torch.compile reduce-overhead)

| Model | Ada (ms) | Spark (ms) | Spark/Ada |
|-------|---------|------------|-----------|
| MobileNetV3-Large | 1.29 | ? | ? |
| EfficientNet-B0 | 1.70 | ? | ? |
| ResNet-152 | 2.57 | ? | ? |

### Batch Scaling Comparison

| B | Ada Custom eager (img/s) | Spark Custom eager (img/s) | Ada gain | Spark gain |
|---|-------------------------|---------------------------|---------|------------|
| 1 | 526 | ? | 1.00× | 1.00× |
| 2 | — | ? | — | ? |
| 4 | 588 | ? | 1.12× | ? |

---

## 8. Execution Order and Time Estimates

| Step | Description | Time | Depends On |
|------|-------------|------|------------|
| 1 | Port --fp16_method | 15 min | — |
| 2 | Re-run ConvMixer model.half() | 20 min | Step 1 |
| 3 | **Batch sweep B=1,2,4** | **30 min** | **Step 1** |
| 4 | Baselines | 20 min | Step 1 |
| 5 | Tile sensitivity | 30 min | — |
| 6 | **L2 NCU extraction** | **15 min** | **— (data exists)** |
| 7 | Cross-platform table | 30 min | Steps 2–6 |

**Total: ~2.5 hours** (Steps 3 and 6 are the additions vs previous plan).

**Recommended order**: Run Step 6 first (fastest, uses existing data, gives immediate L2 insight). Then Steps 1→2→3→4→5→7.

---

## 9. What Can Be Skipped

- **Re-running per-kernel benchmarks**: Method-independent. Existing data correct.
- **Re-running cuDNN layout study**: Method-independent.
- **Re-running Triton baselines**: Method-independent.
- **Re-running compile-survival**: Already model.half(). Correct.
- **Re-running accuracy validation**: Unchanged by FP16 method.
- **Re-running NCU/NSYS from scratch**: Already collected. Just extract L2 metrics (Step 6).

---

## 10. ConvMixer-768/32 (7×7) — The Cleanest L2 Regime Demonstration

This config deserves special attention because it shows the sharpest cross-platform reversal:

| | Ada (96MB L2) | Spark (~24MB L2) |
|---|---|---|
| Per-kernel Custom vs cuDNN | cuDNN NHWC **wins** (18 vs 28 μs) | Custom **wins** (25 vs 362 μs NHWC) |
| E2E Custom+compile(r-o) vs cuDNN | **1.31×** (0.87 vs 1.14 ms) | Expected **larger** |
| Why? | Ada L2 absorbs cuDNN's working set | Spark L2 can't → Custom shmem tiling matters |

**This is the poster child for the L2 residency principle.** A kernel size where cuDNN has a specialized implementation (7×7 NHWC), yet Custom still wins on the small-L2 platform because L2 residency is the governing variable, not library implementation quality.

For the paper: frame the 7×7 result as "K=7 marks the empirical regime boundary where cuDNN's optimized NHWC kernel wins on large-L2 GPUs but loses on small-L2 platforms — the same custom kernel outperforms on both, confirming that L2 residency, not kernel size, is the governing variable."

---

## 11. Output Files Expected

```
results/
├── spark_1536_20_model_half.log           ⬜ (Step 2)
├── spark_768_32_model_half.log            ⬜ (Step 2)
├── spark_batch_b1.log                     ⬜ (Step 3 — NEW)
├── spark_batch_b2.log                     ⬜ (Step 3 — NEW)
├── spark_batch_b4.log                     ⬜ (Step 3 — NEW)
├── spark_baselines_model_half.log         ⬜ (Step 4)
├── spark_sensitivity_31x31.csv            ⬜ (Step 5)
├── spark_sensitivity_31x31.log            ⬜ (Step 5)
├── spark_ncu_l2.csv                       ⬜ (Step 6 — NEW)
├── spark_cross_platform_table.md          ⬜ (Step 7)
│
│ (existing — reusable)
├── spark_hardware_info.txt                ✅
├── spark_convmixer_1536_20.txt            ✅ (autocast — reference only)
├── spark_convmixer_1536_20_v2.txt         ✅ (model.half() compiled)
├── spark_convmixer_768_32.txt             ✅ (autocast — reference only)
├── spark_replknet.txt                     ✅ (model.half())
├── spark_cudnn_layouts.csv                ✅
├── spark_triton_results.csv               ✅
├── spark_kernel_comparison.csv            ✅
├── spark_ncu_metrics.csv                  ✅
├── spark_ncu_tables.md                    ✅
├── spark_roofline_data.csv                ✅
├── spark_roofline_plot.png                ✅
└── ncu/                                   ✅
```

---

## 12. Paper Table Audit — Data Gaps Found (2026-05-10)

Systematic audit of all 18 tables and 5 figures revealed gaps below.
Items marked ⬜ need GPU time on **this machine (Spark/GB10)**.
Items marked 🔧 are data-entry fixes or entries fillable from existing files.

### 12.1 Fillable from Existing Spark Data (🔧, no GPU needed)

The following `tab:kernel_perf` GB10 entries show "---" but data EXISTS in this branch:

**From `results/spark_cudnn_layouts.csv`:**

| Config | cuDNN NCHW (μs) | cuDNN NHWC (μs) | cuDNN-best |
|--------|-----------------|-----------------|------------|
| 7×7, 768ch | 71 | 362 | 71 (NCHW) |
| 9×9, 1536ch | 295 | 1088 | 295 (NCHW) |
| 13×13, 1024ch | 13 | 54 | 13 (NCHW) |
| 27×27, 512ch | 44 | 301 | 44 (NCHW) |
| 29×29, 256ch | 198 | 1026 | 198 (NCHW) |
| 31×31, 128ch | 416 | 2045 | 416 (NCHW) |

**From `results/spark_kernel_comparison.csv` (Custom + Triton):**

| Config | Custom (μs) | Triton (μs) | vs cuDNN-best |
|--------|------------|-------------|---------------|
| 7×7, 768ch | 25 | 96 | 2.84× |
| 9×9, 1536ch | 73 | 299 | 4.04× |
| 13×13, 1024ch | 190 | 16 | 0.07× |
| 27×27, 512ch | 177 | 87 | 0.25× |
| 29×29, 256ch | 268 | 265 | 0.74× |
| 31×31, 128ch | 428 | 717 | 0.97× |

**Decision needed on Triton GB10**: Triton is slower than cuDNN NCHW for K≥27 on GB10. Recommendation: include only for 7×7 and 9×9; mark K≥27 as "---" with footnote explaining autotune failure.

**Decision needed on GB10 13×13**: Custom = 190 μs is 14.6× slower than cuDNN NCHW (13 μs). Tiny spatial dimension (7×7) makes launch overhead dominate. Recommendation: either include honestly or drop 13×13 row.

### 12.2 Missing Experiments on Spark (⬜, need GPU time)

| # | Table | What's Missing | Priority | Time | Why Needed |
|---|-------|---------------|----------|------|------------|
| E1 | `tab:pointwise_layout` | **GB10 pointwise NCHW vs NHWC** for 4 configs | **HIGH** | 10 min | Table 1 is Ada-only but paper emphasizes cross-platform. NHWC speedup expected to be larger on GB10 due to lower BW (273 vs 960 GB/s) |
| E2 | `tab:batch` | **GB10 RepLKNet-31B batch sweep** (B=1,2,4) for cuDNN + Custom | MEDIUM | 30 min | Table only has Ada rows. Would show BW-saturation on GB10 for third architecture |
| E3 | `tab:batch` | **GB10 ConvMixer-768/32 batch sweep** (B=1,2,4) | MEDIUM | 20 min | Same rationale as E2 |
| E4 | `tab:batch` | **GB10 ConvMixer-1536/20 cuDNN-NCHW** batch (B=1,2,4) | LOW | 15 min | Currently only cuDNN-NHWC + Custom columns for GB10. Requires `--cudnn_layout nchw` flag |
| E5 | `tab:batch` | GB10 Triton batch sweep | **SKIP** | — | Triton 5.81× slower than Custom on GB10. Not meaningful data |

### 12.3 Experiment Commands

**E1 — Pointwise Layout Benchmark (10 min):**
```bash
python benchmark_pointwise_layout.py --device cuda:0 --num_runs 500 \
    --output results/spark_pointwise_layout.csv
```
Script is in repo root. Measures 1×1 conv NCHW vs NHWC for 4 configs.
**Expected**: NHWC speedup larger than Ada (1.18–1.22×) because BW is scarcer → layout matters more.

**E2 — RepLKNet-31B Batch Sweep (30 min):**
```bash
for B in 1 2 4; do
    python replknet/benchmark_replknet.py --device cuda:0 --batch $B \
        --num_runs 200 --fp16_method model_half \
        2>&1 | tee results/spark_replknet_batch_b${B}.log
done
```
**Expected**: Custom throughput flat/declining (already BW-saturated at B=1 per §3 results).

**E3 — ConvMixer-768/32 Batch Sweep (20 min):**
```bash
for B in 1 2 4; do
    python convmixer/benchmark.py --model 768_32 --skip_accuracy \
        --device cuda:0 --fp16_method model_half --batch $B --real_data \
        --num_runs 200 \
        2>&1 | tee results/spark_768_32_batch_b${B}.log
done
```

**E4 — ConvMixer-1536/20 cuDNN-NCHW Batch (15 min):**
```bash
for B in 1 2 4; do
    python convmixer/benchmark.py --model 1536_20 --skip_accuracy \
        --device cuda:0 --fp16_method model_half --batch $B --real_data \
        --cudnn_layout nchw --num_runs 200 \
        2>&1 | tee results/spark_1536_20_cudnn_nchw_batch_b${B}.log
done
```
Note: requires `--cudnn_layout nchw` flag. Check if already ported to SPARK branch.

### 12.4 Output Files Expected

```
results/
├── spark_pointwise_layout.csv               ⬜ (E1 — NEW)
├── spark_replknet_batch_b1.log              ⬜ (E2 — NEW)
├── spark_replknet_batch_b2.log              ⬜ (E2 — NEW)
├── spark_replknet_batch_b4.log              ⬜ (E2 — NEW)
├── spark_768_32_batch_b1.log                ⬜ (E3 — NEW)
├── spark_768_32_batch_b2.log                ⬜ (E3 — NEW)
├── spark_768_32_batch_b4.log                ⬜ (E3 — NEW)
├── spark_1536_20_cudnn_nchw_batch_b1.log    ⬜ (E4 — NEW)
├── spark_1536_20_cudnn_nchw_batch_b2.log    ⬜ (E4 — NEW)
└── spark_1536_20_cudnn_nchw_batch_b4.log    ⬜ (E4 — NEW)
```

### 12.5 Priority Order

1. **E1**: Pointwise layout benchmark (HIGH — closes biggest cross-platform gap)
2. **E2**: RepLKNet batch sweep (MEDIUM — fills tab:batch GB10 gap for third architecture)
3. **E3**: ConvMixer-768/32 batch sweep (MEDIUM — second architecture batch data)
4. **E4**: cuDNN-NCHW batch sweep (LOW — best-cuDNN comparison)

Total estimated GPU time: **~75 minutes**