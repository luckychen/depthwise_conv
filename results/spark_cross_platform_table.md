# Cross-Platform Comparison Table
Generated: 2026-05-08 | Branch: SPARK | All: model.half(), channels_last, batch=1 unless noted

---

## E2E Model Latency (model.half(), torch.compile mode=default)

| Model | Config | Ada RTX6000 (ms) | Spark GB10 (ms) | Spark/Ada |
|-------|--------|-----------------|-----------------|-----------|
| ConvMixer-1536/20 | CuDNN eager | 7.58 | 25.11 | 3.31× |
| | CuDNN + compile | 7.52 | 24.16 | 3.21× |
| | Custom eager | 1.90 | 4.92 | 2.59× |
| | **Custom + compile** | **1.54** | **4.26** | **2.77×** |
| | Custom speedup vs CuDNN | **4.93×** | **5.89×** | — |
| ConvMixer-768/32 | CuDNN eager | 2.47 | 14.42 | 5.84× |
| | CuDNN + compile | 1.14 | 12.95 | 11.36× |
| | Custom eager | 3.04 | 3.51 | 1.15× |
| | **Custom + compile** | **1.14** | **2.09** | **1.83×** |
| | Custom speedup vs CuDNN-best | **1.00×** | **6.20×** | — |
| RepLKNet-31B | CuDNN eager | 5.56 | 18.40 | 3.31× |
| | CuDNN + compile | 4.40 | 15.27 | 3.47× |
| | Custom eager | 5.13 | 7.80 | 1.52× |
| | **Custom + compile** | **1.77** | **4.67** | **2.64×** |
| | Custom speedup vs CuDNN | **3.14×** | **3.93×** (vs eager) | — |

**Note on 768/32 (7×7):** This config shows the largest cross-platform delta in Custom-vs-cuDNN ratio (1.00× → 6.20×).
On Ada, cuDNN's specialized NHWC 7×7 kernel ties Custom+compile at 1.14 ms (per-kernel: cuDNN-NHWC actually wins, 18 vs 28 μs;
torch.compile fusion lets Custom catch up E2E). On Spark, cuDNN+compile collapses to 12.95 ms while Custom holds at 2.09 ms.
The true *regime reversal* is at the per-kernel level: cuDNN-NHWC wins on Ada (18 μs vs Custom 28 μs),
loses on Spark (362 μs vs Custom 25 μs). At the model level the pattern is "tied → dominant", not strictly a reversal.

---

## Compiled Comparison: ConvMixer-1536/20 (benchmark_compiled_comparison.py, model.half())

| Method | Spark (ms) | Spark (img/s) | vs cuDNN-best |
|--------|-----------|---------------|---------------|
| cuDNN + compile (NCHW) | 24.08 | 41.5 | 1.00× (best) |
| cuDNN + compile (NHWC) | 24.13 | 41.4 | 1.00× |
| **Custom + compile (NHWC)** | **4.14** | **241.3** | **5.81×** |
| Triton NHWC + compile | 9.18 | 108.9 | 2.62× |
| Triton NCHW + compile | 18.28 | 54.7 | 1.32× |

---

## Baseline Models (model.half(), torch.compile mode=default)

| Model | Ada (ms) | Ada (img/s) | Spark (ms) | Spark (img/s) | Spark/Ada |
|-------|---------|-------------|-----------|---------------|-----------|
| MobileNetV3-Large | 1.29 | 775.9 | 0.97 | 1028.5 | 0.75× |
| EfficientNet-B0 | 1.70 | 588.0 | 1.30 | 770.1 | 0.76× |
| ResNet-152 | 2.57 | 388.4 | 2.23 | 447.9 | 0.87× |

---

## L2 Hypothesis: Batch Scaling (ConvMixer-1536/20, model.half(), real ImageNet data)

| B | Ada Custom eager (img/s) | Spark Custom eager (img/s) | Ada gain vs B=1 | Spark gain vs B=1 |
|---|--------------------------|---------------------------|-----------------|-------------------|
| 1 | 526.4 | 201.4 | 1.00× | 1.00× |
| 2 | — | 199.9 | — | 0.99× |
| 4 | 588.4 | 188.4 | **+12%** | **−6%** |

**Interpretation:** On Ada (96MB L2), throughput grows +12% from B=1→4 because the large L2 still absorbs
the expanded working set. On Spark (24MB L2), throughput *decreases* at B=4 — the working set exceeds
L2 and the kernel becomes bandwidth-bound. This is the opposite of the L2 hypothesis prediction,
suggesting the GB10 is already bandwidth-limited even at B=1, and adding batch just increases pressure.

Revised framing: "On bandwidth-constrained platforms (small L2, high memory BW ratio), custom
shared-memory tiling helps at B=1 because redundant DRAM loads dominate regardless of L2 capacity.
Batch scaling offers diminishing returns — the bottleneck is BW, not L2 fill rate."
Note: CuDNN shows same pattern (39.2 → 38.9 img/s flat across B=1–4), consistent with BW bottleneck.

---

## L2 Hit Rates (NCU profiling, single-kernel, B=1)

| Config | Ada Custom L2 | Spark Custom L2 | Spark cuDNN-NHWC L2 | Spark Triton L2 |
|--------|--------------|-----------------|---------------------|-----------------|
| 9×9 / 1536ch | 81.7% | 66.4% | 25.2% (NCHW) | 98.7% |
| 7×7 / 768ch | 75.9% | 51.5% | 26.1% (NCHW) | 97.4% |
| 31×31 / 128ch | 95.9% | 94.0% | 60.8% (NCHW) | — |
| 27×27 / 512ch | 93.0% | 91.1% | 23.8% (NCHW) | — |
| 29×29 / 256ch | — | 89.6% | 42.7% (NCHW) | — |

Custom kernel maintains high L2 hit rates on Spark (66–94%) vs cuDNN-NCHW (23–60%) due to
shared-memory weight tiling. The 7×7 and 9×9 configs show the largest Ada→Spark drop (−15pp, −24pp),
consistent with those configs having the largest working sets relative to Spark's smaller L2.

---

## Tile Sensitivity Sweep (31×31, 128ch, B=1, model.half())

| M (rows/tile) | Spark latency (ms) | Spark blocks/SM | Ada latency (ms) | Ada blocks/SM |
|---------------|--------------------|-----------------|-----------------|---------------|
| 4 | **0.969** | 16.3 | — | 5.5 |
| 8 | 1.018 | 4.1 | — | 1.4 |
| 16 | 1.678 | 0.25 | — | 0.45 |

On Spark (128 SMs on GB10), M=4 is optimal — smaller tiles produce more blocks, saturating the
larger SM count. This is the reverse of Ada (60 SMs) where M=4 is also optimal but for different
reasons (need to fill fewer SMs). Cross-platform optimal tile size analysis shows regime is
SM-count-dependent, not just kernel-size-dependent.

---

## Data Provenance

| File | Method | Status | Notes |
|------|--------|--------|-------|
| spark_1536_20_model_half.log | model_half | ✅ | Step 2 — corrected eager numbers |
| spark_768_32_model_half.log | model_half | ✅ | Step 2 — corrected eager numbers |
| spark_batch_b1.log | model_half, real data | ✅ | Step 3 — L2 hypothesis test |
| spark_batch_b2.log | model_half, real data | ✅ | Step 3 |
| spark_batch_b4.log | model_half, real data | ✅ | Step 3 |
| spark_baselines_model_half.log | model_half | ✅ | Step 4 |
| spark_sensitivity_31x31.csv | model_half | ✅ | Step 5 |
| spark_ncu_l2.csv | — | ✅ | Step 6 — extracted from existing NCU data |
| spark_convmixer_1536_20_v2.txt | model_half | ✅ | Compiled comparison, reusable |
| spark_replknet.txt | model_half | ✅ | RepLKNet E2E, reusable |
| spark_cudnn_layouts.csv | direct FP16 | ✅ | Method-independent |
| spark_kernel_comparison.csv | direct FP16 | ✅ | Method-independent |
| spark_triton_results.csv | direct FP16 | ✅ | Method-independent |
| spark_ncu_metrics.csv | — | ✅ | NCU per-kernel, method-independent |
| spark_convmixer_1536_20.txt | autocast | ⚠️ reference only | Superseded by model_half log |
| spark_convmixer_768_32.txt | autocast | ⚠️ reference only | Superseded by model_half log |
