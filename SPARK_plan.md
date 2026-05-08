# DGX Spark — Remaining Experiment Plan

**Date**: 2026-05-08 (updated — expanded to all 3 models per Ada findings)
**Status**: ⚠ Batch sweep needs cuDNN-NCHW + Triton for ALL 3 models
**Branch**: SPARK
**Context**: `SPARK_context.md` — read for full background.
**Pair**: `RTX6000_Ada_plan.md` on main branch

---

## 0. Critical New Finding from Ada (2026-05-08 Full Batch Sweep)

The Ada batch sweep across all 4 methods × 3 models revealed a cross-model pattern:

> **cuDNN-NCHW (compile-decomposed to 768+ GEMMs) scales strongly with batch — but the magnitude
> differs drastically by model.** 1536/20: 135→313 img/s (2.3× gain), 768/32: 770→793 (flat),
> RepLKNet: 207→881 (4.3× gain). The difference tracks working-set size: larger models benefit
> more from GEMM batch efficiency because each decomposed GEMM has larger M/N/K dimensions.
> Custom's advantage vs cuDNN-NCHW narrows from 4.92× (B=1) to 1.95× (B=4) for 1536/20, and
> from 1.91× to 1.02× for RepLKNet.

**The Spark question**: On a BW-constrained platform (~24 MB L2, ~273 GB/s), do these
compile-decomposed GEMMs also scale, or does BW pressure cap them? The existing Spark batch
data only covers 1536/20 Custom + cuDNN-NHWC. We need cuDNN-NCHW + Triton for all three models
to test the cross-platform hypothesis.

---

## 1. Previously Completed Spark Work

| ID | Description | Status |
|----|-------------|--------|
| S1 | Port --fp16_method to convmixer/benchmark.py | ✅ commit a82dc34 |
| S2 | Re-run ConvMixer 1536/20 + 768/32 with model.half() | ✅ |
| S3 | Batch sweep B=1,2,4: Custom + cuDNN-NHWC, 1536/20 only | ⚠ Needs expansion |
| S4 | Baseline models (MobileNetV3, EffNet, ResNet) | ✅ |
| S5 | Tile sensitivity (31×31/128ch) | ✅ |
| S6 | L2 hit rates from NCU data | ✅ |
| S7 | Cross-platform comparison table | ⚠ Needs update |

---

## 2. NEW EXPERIMENTS — Full Cross-Model Batch Sweep

### 2.1 cuDNN-NCHW Batch Scaling — All 3 Models (~45 min)

The single most important remaining experiment. On Ada, cuDNN-NCHW batch scaling varies from
flat (768/32) to 2.3× (1536/20) to 4.3× (RepLKNet). On Spark, BW saturation should compress
this range — prediction: all models flat or weakly scaling.

**ConvMixer-1536/20 (9×9, 20 layers, 51.6M params):**
```bash
for B in 1 2 4; do
    python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
        --device cuda:0 --fp16_method model_half --batch $B --real_data \
        --data_dir ./data/imagenet --cudnn_layout nchw \
        2>&1 | tee results/spark_batch_cudnn_nchw_1536_b${B}.log
done
```

**ConvMixer-768/32 (7×7, 32 layers, 21.1M params):**
```bash
for B in 1 2 4; do
    python convmixer/benchmark.py --model 768_32 --skip_accuracy --num_runs 200 \
        --device cuda:0 --fp16_method model_half --batch $B --real_data \
        --data_dir ./data/imagenet --cudnn_layout nchw \
        2>&1 | tee results/spark_batch_cudnn_nchw_768_b${B}.log
done
```

**RepLKNet-31B (27-31×31, 24 layers, 79M params):**
```bash
for B in 1 2 4; do
    python replknet/benchmark.py --device cuda:0 --num_runs 200 \
        --fp16_method model_half --cudnn_layout nchw --batch $B \
        --data_dir ./data/imagenet --real_data \
        2>&1 | tee results/spark_batch_cudnn_nchw_replknet_b${B}.log
done
```

### 2.2 Triton Batch Scaling — 1536/20 + 768/32 (~25 min)

RepLKNet Triton batch is skipped — Ada showed autotune issue for RepLKNet E2E.

```bash
for B in 1 2; do
    python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
        --device cuda:0 --fp16_method model_half --batch $B --real_data \
        --data_dir ./data/imagenet --triton_mode \
        2>&1 | tee results/spark_batch_triton_1536_b${B}.log

    python convmixer/benchmark.py --model 768_32 --skip_accuracy --num_runs 200 \
        --device cuda:0 --fp16_method model_half --batch $B --real_data \
        --data_dir ./data/imagenet --triton_mode \
        2>&1 | tee results/spark_batch_triton_768_b${B}.log
done
```

### 2.3 Custom + cuDNN-NHWC Batch Baseline — 768/32 (~15 min)

Existing Spark batch sweep only covers 1536/20. Fill in 768/32:

```bash
for B in 1 2 4; do
    python convmixer/benchmark.py --model 768_32 --skip_accuracy --num_runs 200 \
        --device cuda:0 --fp16_method model_half --batch $B --real_data \
        --data_dir ./data/imagenet \
        2>&1 | tee results/spark_batch_baseline_768_b${B}.log
done
```

---

## 3. Expected Cross-Platform Findings

| Model + Method | Ada pattern (96 MB, ~960 GB/s) | Predicted Spark pattern (~24 MB, ~273 GB/s) |
|----------------|-------------------------------|------------------------------------------|
| 1536/20 cuDNN-NCHW | Strong scaling (135→313, 2.3×) | **Flat** — GEMMs BW-saturated at B=1 |
| 768/32 cuDNN-NCHW | Flat (770→793) | **Flat** — already compute-light |
| RepLKNet cuDNN-NCHW | Strong scaling (207→881, 4.3×) | **Weak or flat** — largest working set, most BW pressure |
| 768/32 cuDNN-NHWC | Strong scaling (777→1844, 2.4×) | **Weak** — specialized 7×7 kernel BW-capped |
| Custom all models | Flat/declining | **Flat/declining** — BW-saturated at B=1 |
| Triton all models | Flat/declining | **Flat/declining** — no cross-program shmem |

### Refined cross-platform claim

> **On Spark, all methods are BW-capped at B=1. Custom dominates through shmem weight
> caching (5.89×-6.20× E2E). cuDNN-NCHW's compile-decomposed GEMMs cannot scale with
> batch. On Ada, GEMM batch efficiency allows cuDNN-NCHW to partially recover at B≥4,
> but Custom still leads at B=1 — the deployment target for latency-sensitive inference.
> The governing variable is DRAM bandwidth pressure, with L2 hit rate as the tracker.**

This is the cleanest version of the paper's thesis: Custom wins where BW matters; cuDNN-NCHW
recovers only where BW is plentiful; the cross-platform magnitude difference IS the regime.

---

## 4. Execution Order and Time Estimates

| Step | Description | Time |
|------|-------------|------|
| 1 | Cherry-pick Ada batch-sweep commits (3f8ad12, ad55f01, 163920c) | 15 min |
| 2 | cuDNN-NCHW batch: 1536/20 B=1,2,4 | 15 min |
| 3 | cuDNN-NCHW batch: 768/32 B=1,2,4 | 15 min |
| 4 | cuDNN-NCHW batch: RepLKNet B=1,2,4 | 15 min |
| 5 | Triton batch: 1536/20 + 768/32 B=1,2 | 25 min |
| 6 | 768/32 baseline (Custom + cuDNN-NHWC) B=1,2,4 | 15 min |
| 7 | Build cross-platform batch comparison table | 20 min |

**Total: ~2 hours**

---

## 5. What Can Be Skipped

- Per-kernel benchmarks, cuDNN layout study, Triton baselines — method-independent, reusable
- Compile-survival, NCU, roofline — already collected, reusable
- Accuracy validation — unchanged
- RepLKNet Triton batch — Ada confirmed autotune issue
- Spark batch B=8 — Ada shows no additional pattern beyond B=4; B=1,2,4 is sufficient

---

## 6. Output Files Expected

```
results/
├── spark_batch_cudnn_nchw_1536_b1–b4.log      ⬜ §2.1
├── spark_batch_cudnn_nchw_768_b1–b4.log       ⬜ §2.1
├── spark_batch_cudnn_nchw_replknet_b1–b4.log  ⬜ §2.1
├── spark_batch_triton_1536_b1–b2.log          ⬜ §2.2
├── spark_batch_triton_768_b1–b2.log           ⬜ §2.2
├── spark_batch_baseline_768_b1–b4.log         ⬜ §2.3
├── spark_cross_platform_batch_table.csv       ⬜ §7
│
│ (existing — reusable)
├── spark_1536_20_model_half.log               ✅
├── spark_768_32_model_half.log                ✅
├── spark_batch_b1–b4.log                      ✅ (1536/20: Custom + cuDNN-NHWC)
├── spark_baselines_model_half.log             ✅
├── spark_sensitivity_31x31.csv                ✅
├── spark_ncu_l2.csv                           ✅
├── spark_cudnn_layouts.csv                    ✅
├── spark_triton_results.csv                   ✅
├── spark_kernel_comparison.csv                ✅
├── spark_ncu_metrics.csv                      ✅
├── spark_roofline_plot.png                    ✅
└── ncu/                                       ✅
```