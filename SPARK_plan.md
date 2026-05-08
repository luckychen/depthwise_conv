# DGX Spark — Experiment Plan

**Date**: 2026-05-08 (updated — batch methodology gap found, new experiments added)
**Status**: ⚠ MOSTLY COMPLETE — batch sweep needs cuDNN-NCHW and Triton columns
**Branch**: SPARK
**Context**: `SPARK_context.md` — read for full background.
**Pair**: `RTX6000_Ada_plan.md` on main branch

---

## 0. Headline Findings (Completed Work)

1. **Custom dominates on Spark across all configs.** ConvMixer-1536/20 Custom+compile = 4.26 ms vs
   cuDNN+compile = 24.16 ms (**5.67×**). ConvMixer-768/32 Custom+compile = 2.09 ms vs cuDNN+compile =
   12.95 ms (**6.20×**). RepLKNet-31B Custom+compile = 4.67 ms vs cuDNN+compile = 15.27 ms (**3.27×**).

2. **The L2 batch hypothesis was disconfirmed but yields a sharper claim.** Spark Custom eager
   throughput: B=1 → 201.4, B=2 → 199.9, B=4 → 188.4 img/s. Throughput *decreases* with batch — the
   GB10 is bandwidth-saturated even at B=1. Revised framing: **shared-memory tiling pays off when
   DRAM bandwidth — not arithmetic — is the bottleneck.** The original L2-capacity-threshold form
   was tested and refined into this broader BW-pressure form.

3. **L2 hit rates quantitatively confirm the shmem-tiling story.** Custom holds 51--94% L2 hit on
   Spark vs cuDNN-NCHW at 23--60%. Output: `results/spark_ncu_l2.csv`.

4. **Per-kernel 7×7/768ch is the true regime reversal**: cuDNN-NHWC wins on Ada (18 vs 28 μs),
   loses badly on Spark (362 vs 25 μs). At E2E, torch.compile fusion lifts Custom to a tie with
   cuDNN on Ada (1.14 ms each), then dominates 6.20× on Spark.

5. **Tile sensitivity (31×31/128ch) on Spark**: M=4 → 0.969 ms, M=8 → 1.018 ms, M=16 → 1.678 ms.
   M=4 wins; optimal tile is SM-count-dependent.

---

## 1. Critical Constraints

The SPARK branch carries fixes for CUDA stream (CUDA-graph compatibility) and layout conversion
(remove permute/contiguous). **Do NOT revert these.** Clear stale JIT cache before running:

```bash
rm -rf ~/.cache/torch_extensions/*/depthwise_convmixer
```

Timing: use **per-iteration CUDA events** for all E2E benchmarks. Bulk timing is wrong for
`reduce-overhead` mode (up to 4× error).

---

## 2. Completed Experiments

| Section | Description | Status |
|---------|-------------|--------|
| S1 | Port --fp16_method to convmixer/benchmark.py | ✅ commit a82dc34 |
| S2 | Re-run ConvMixer 1536/20 + 768/32 with model.half() | ✅ |
| S3 | Batch sweep B=1,2,4 — Custom + cuDNN-**NHWC** | ⚠ cuDNN-NCHW missing (see §3) |
| S4 | Baseline models (MobileNetV3, EffNet, ResNet) | ✅ |
| S5 | Tile sensitivity sweep | ✅ |
| S6 | L2 hit rates from existing NCU data | ✅ |
| S7 | Cross-platform comparison table | ✅ |

---

## 3. NEW EXPERIMENTS — Batch Sweep with cuDNN-NCHW and Triton

### Why These Are Needed (C0 from DS_suggestion.md)

The paper's stated methodology is: *"We report the best cuDNN configuration as the baseline."*
The B=1 E2E tables follow this correctly. But the batch sweep in §5.6 silently uses cuDNN-NHWC
(39.3 img/s at B=1), not cuDNN's best layout (NCHW). cuDNN-NCHW at B=1 is likely several times
faster — we don't know because it was never measured.

Additionally, Triton has **zero** batch scaling data on Spark. If Triton scales better with batch
than Custom does, the "Triton collapse" finding only holds at B=1.

### 3.1 Measure cuDNN-NCHW Batch Scaling (~20 min)

ConvMixer-1536/20, model.half(), real ImageNet, mode=default compile:

```bash
# Need to port the --cudnn_layout nchw flag from Ada's convmixer/benchmark.py first,
# OR manually construct an NCHW model and run the throughput function.

# B=1
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 1 --real_data \
    --data_dir ./data/imagenet --cudnn_layout nchw \
    2>&1 | tee results/spark_batch_cudnn_nchw_b1.log

# B=2
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 2 --real_data \
    --data_dir ./data/imagenet --cudnn_layout nchw \
    2>&1 | tee results/spark_batch_cudnn_nchw_b2.log

# B=4
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 4 --real_data \
    --data_dir ./data/imagenet --cudnn_layout nchw \
    2>&1 | tee results/spark_batch_cudnn_nchw_b4.log
```

**If `--cudnn_layout nchw` doesn't exist yet**: Port it from Ada's `convmixer/benchmark.py` (cherry-pick the relevant commit from main), or manually modify the benchmark to skip channels_last conversion and pass NCHW tensors.

Record: latency (ms) and throughput (img/s) at B=1, 2, 4.

### 3.2 Measure Triton Batch Scaling (~20 min)

ConvMixer-1536/20 with Triton depthwise, model.half(), real ImageNet:

```bash
# If a --triton_mode flag exists (port from Ada if needed):
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 1 --real_data \
    --data_dir ./data/imagenet --triton_mode \
    2>&1 | tee results/spark_batch_triton_b1.log

python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 2 --real_data \
    --data_dir ./data/imagenet --triton_mode \
    2>&1 | tee results/spark_batch_triton_b2.log
```

Record: latency and throughput at B=1, B=2.

### 3.3 Expected Findings

| Scenario | Interpretation |
|----------|---------------|
| cuDNN-NCHW scales similarly to NHWC (flat/declining) | BW-saturation confirmed across both cuDNN layouts |
| cuDNN-NCHW scales BETTER with batch | Paper must acknowledge cuDNN's best layout mitigates BW pressure at batch>1 |
| Triton throughput flat/declining | "Triton collapse" holds at all batch sizes |
| Triton throughput IMPROVES with batch | Paper must acknowledge Triton recovers at batch>1 |

### 3.4 After Measuring

Update `results/spark_cross_platform_table.md` with the new batch columns.
Update paper.tex Table `tab:batch` to include cuDNN-NCHW and Triton columns for both platforms.

---

## 4. What Was Skipped (Existing Data Reusable)

- Per-kernel benchmarks (`spark_kernel_comparison.csv`) — method-independent, reusable.
- cuDNN layout study (`spark_cudnn_layouts.csv`) — direct FP16 tensors, reusable.
- Triton baselines (`spark_triton_results.csv`) — direct FP16 tensors, reusable.
- Compile-survival / NSYS — already model.half(), reusable.
- NCU per-kernel profiling — already collected, reusable.
- Roofline plot — derived from NCU data, reusable.
- Accuracy validation — unchanged by FP16 method.

---

## 5. ConvMixer-768/32 (7×7) — Corrected Framing

| Level | Ada | Spark | Pattern |
|---|---|---|---|
| Per-kernel 7×7/768ch | cuDNN-NHWC **wins** (18 vs 28 μs) | Custom **wins** (25 vs 362 μs NHWC) | **True regime reversal** |
| E2E model 768/32 (compile) | Tie at 1.14 ms each | Custom 2.09 vs cuDNN 12.95 — **6.20×** | "Tied → dominant" |

**Recommended paper wording**: *"At the per-kernel level, K=7 marks an L2-driven regime
reversal: cuDNN's specialized NHWC 7×7 kernel wins on the 96 MB-L2 Ada by 1.6× but loses on
the 24 MB-L2 Spark by 14.5×. At the model level, torch.compile fusion equalizes Custom and
cuDNN on Ada (tied at 1.14 ms), but Spark's L2 cannot absorb cuDNN's working set under the
compile schedule — Custom wins by 6.20×, the largest cross-platform Custom-vs-cuDNN delta in
the dataset."*

---

## 6. Execution Order

| Step | Description | Time | Depends On |
|------|-------------|------|------------|
| 1 | Port --cudnn_layout nchw and --triton_mode from Ada benchmark.py | 15 min | — |
| 2 | cuDNN-NCHW batch sweep B=1,2,4 | 20 min | Step 1 |
| 3 | Triton batch sweep B=1,2 | 20 min | Step 1 |
| 4 | Update cross-platform table | 15 min | Steps 2-3 |

**Total: ~1 hour.**

---

## 7. Output Files

```
results/
├── spark_1536_20_model_half.log             ✅
├── spark_768_32_model_half.log              ✅
├── spark_batch_b1.log                       ✅ (Custom + cuDNN-NHWC)
├── spark_batch_b2.log                       ✅ (Custom + cuDNN-NHWC)
├── spark_batch_b4.log                       ✅ (Custom + cuDNN-NHWC)
├── spark_batch_cudnn_nchw_b1.log            ⬜ §3.1 — NEW
├── spark_batch_cudnn_nchw_b2.log            ⬜ §3.1 — NEW
├── spark_batch_cudnn_nchw_b4.log            ⬜ §3.1 — NEW
├── spark_batch_triton_b1.log                ⬜ §3.2 — NEW
├── spark_batch_triton_b2.log                ⬜ §3.2 — NEW
├── spark_baselines_model_half.log           ✅
├── spark_sensitivity_31x31.csv              ✅
├── spark_ncu_l2.csv                         ✅
├── spark_cross_platform_table.md            ⚠ Needs update after §3
│
│ (reusable)
├── spark_cudnn_layouts.csv                  ✅
├── spark_triton_results.csv                 ✅
├── spark_kernel_comparison.csv              ✅
├── spark_ncu_metrics.csv                    ✅
├── spark_roofline_plot.png                  ✅
├── ncu/                                     ✅
│
│ (autocast-era — reference only)
├── spark_convmixer_1536_20.txt              ⚠ reference only
└── spark_convmixer_768_32.txt               ⚠ reference only
```