# DGX Spark — Experiment Plan (COMPLETED)

**Date**: 2026-05-08
**Status**: ✅ ALL EXPERIMENTS COMPLETE. No further Spark work needed.
**Branch**: SPARK
**Context**: `SPARK_context.md` — read for full background.
**Pair**: `RTX6000_Ada_plan.md` on main branch

**Scope decision (2026-05-08)**: Spark is the **latency-oriented edge platform** — B=1 only.
Batch scaling is evaluated on Ada (throughput-oriented desktop GPU) where it addresses
reviewer feedback (R1.3, R2.2). Spark B=1 already carries the core cross-platform evidence
(5.89×-6.20× E2E Custom vs cuDNN). NCU L2 data (51-94% Custom vs 23-60% cuDNN-NCHW)
confirms BW saturation at B=1 without needing batch sweeps.

---

## 0. Headline Findings

1. **Custom dominates on Spark across all configs.** ConvMixer-1536/20 Custom+compile = 4.26 ms vs
   cuDNN+compile = 24.16 ms (**5.67×**). ConvMixer-768/32 Custom+compile = 2.09 ms vs cuDNN+compile =
   12.95 ms (**6.20×**). RepLKNet-31B Custom+compile = 4.67 ms vs cuDNN+compile = 15.27 ms (**3.27×**).

2. **BW-pressure governs the regime on this platform.** Spark Custom eager throughput for 1536/20:
   B=1 → 201.4 img/s. The GB10 is bandwidth-saturated even at B=1 (verified by NCU MemBW%).
   Shared-memory tiling reduces redundant DRAM weight loads from the first sample — the advantage
   does not require batch to trigger.

3. **L2 hit rates quantitatively confirm the shmem-tiling story.** Custom holds 51-94% L2 hit on
   Spark vs cuDNN-NCHW at 23-60%. Output: `results/spark_ncu_l2.csv`.

4. **Per-kernel 7×7/768ch is the true regime reversal**: cuDNN-NHWC wins on Ada (18 vs 28 μs),
   loses badly on Spark (362 vs 25 μs). At E2E, torch.compile fusion lifts Custom to a tie with
   cuDNN on Ada (1.14 ms each), then dominates 6.20× on Spark.

5. **Tile sensitivity (31×31/128ch) on Spark**: M=4 → 0.969 ms, M=8 → 1.018 ms, M=16 → 1.678 ms.
   M=4 wins; optimal tile is SM-count-dependent.

---

## 1. Completed Experiments

| ID | Description | Status |
|----|-------------|--------|
| S1 | Port --fp16_method to convmixer/benchmark.py | ✅ commit a82dc34 |
| S2 | Re-run ConvMixer 1536/20 + 768/32 with model.half() | ✅ |
| S3 | Batch sweep B=1,2,4: Custom + cuDNN-NHWC, 1536/20 | ✅ disconfirmed L2-capacity → reframed BW-pressure |
| S4 | Baseline models (MobileNetV3, EffNet, ResNet) | ✅ |
| S5 | Tile sensitivity (31×31/128ch) | ✅ M=4 wins (0.969 ms) |
| S6 | L2 hit rates from NCU data | ✅ Custom 51-94% vs cuDNN-NCHW 23-60% |
| S7 | Cross-platform comparison table | ✅ `results/spark_cross_platform_table.md` |

---

## 2. Why Spark Stays B=1 (Design Decision)

| Reason | Detail |
|--------|--------|
| **Deployment role** | Spark is the latency-oriented edge platform — B=1 is the natural deployment mode |
| **Ada covers batch** | Ada batch sweep (all 4 methods × 3 models × B=1-8) addresses R1.3/R2.2 complaints |
| **NCU already proves BW saturation** | Custom 51-94% L2 hit vs cuDNN-NCHW 23-60% at B=1 already quantifies the BW-pressure regime |
| **Adding batch would not change the thesis** | The paper claims Custom wins through shmem tiling at B=1 — Spark batch would only confirm flat throughput (already evident from B=1 NCU) |

---

## 3. Output Files — Final State

```
results/
├── spark_1536_20_model_half.log               ✅
├── spark_768_32_model_half.log                ✅
├── spark_batch_b1.log                         ✅ (1536/20: Custom + cuDNN-NHWC)
├── spark_batch_b2.log                         ✅
├── spark_batch_b4.log                         ✅
├── spark_baselines_model_half.log             ✅
├── spark_sensitivity_31x31.csv                ✅
├── spark_ncu_l2.csv                           ✅
├── spark_cross_platform_table.md              ✅
├── spark_cudnn_layouts.csv                    ✅
├── spark_triton_results.csv                   ✅
├── spark_kernel_comparison.csv                ✅
├── spark_ncu_metrics.csv                      ✅
├── spark_roofline_plot.png                    ✅
└── ncu/                                       ✅
```