# RTX 6000 Ada Experiment Plan — Cross-Platform Pair for Revision

**Date**: 2026-05-08 (updated — batch methodology gap found, new experiments added)
**Branch**: main
**Pair**: `SPARK_plan.md` on SPARK branch

This file is the executable instruction sheet for the RTX 6000 Ada workstation. It pairs with DGX Spark experiments (`SPARK` branch, `SPARK_context.md`) so the revised paper presents a two-platform BW-pressure regime study.

---

## 0. Central Thesis (Refined 2026-05-08)

> **Shared-memory weight tiling pays off when DRAM bandwidth — not arithmetic — is the bottleneck.**
> L2 hit rate is the quantitative tracker. Cross-platform validation: Ada (96 MB L2, 960 GB/s) vs Spark (~24 MB L2, ~273 GB/s).

The original "L2 capacity threshold" form of this thesis was tested by a batch sweep and refined — see §4.7.2. The governing variable is **bandwidth pressure**, not L2 fill state. Shared-memory tiling reduces redundant DRAM weight loads regardless of whether the working set nominally fits in L2.

---

## 1. Hardware Specs

| Property | Ada (this machine) | Spark |
|---|---|---|
| GPU | NVIDIA RTX 6000 Ada | DGX Spark (GB10) |
| Architecture | Ada Lovelace (SM 8.9) | Blackwell (SM 12.1) |
| SM Count | 142 | 48 |
| L2 Cache | **96 MB** | **~24 MB** (4.4× smaller) |
| Memory | 48 GB GDDR6 | 128 GB unified |
| Memory Bandwidth | 960 GB/s | ~273 GB/s LPDDR5X |
| FP16 Peak | 91.1 TFLOPS | ~209 TFLOPS |

---

## 2. Environment

```bash
source /home/chenchon/miniconda3/etc/profile.d/conda.sh && conda activate cuda
export CUDA_VISIBLE_DEVICES=1
cd /home/chenchon/chConv/paper/latex/repo
```

Stack: Python 3.10, PyTorch 2.9.1+cu128, CUDA 12.8 (nvcc 12.9), cuDNN 9.10.2.

### FP16 Methodology

**Use `model.half()` as the primary methodology.** Autocast kept as secondary column for paper continuity.

| Method | How | When |
|--------|-----|------|
| **model.half()** (primary) | `model = model.half()` then `model(x.half())` | All revision numbers |
| autocast (secondary) | FP32 model, `torch.amp.autocast` per forward | Paper continuity only |

---

## 3. Completed Experiments

### 3.1 Per-Kernel Depthwise — ✅

| Config | Custom (μs) | cuDNN NCHW (μs) | cuDNN NHWC (μs) | Triton (μs) | Custom vs best cuDNN |
|--------|------------|-----------------|-----------------|-------------|---------------------|
| 9×9, 1536ch, 32×32 | 28.1 | 90 | 293 | 127 | **3.2×** (vs NCHW) |
| 7×7, 768ch, 32×32 | 27.7 | 27 | **18** | 43 | 0.65× (cuDNN NHWC wins) |
| 31×31, 128ch, 56×56 | 116 | **150** | 634 | 286 | 1.3× |
| 29×29, 256ch, 28×28 | 72 | **77** | 267 | 128 | 1.07× |
| 27×27, 512ch, 14×14 | 22 | **15** | 93 | 38 | 0.68× (cuDNN NCHW wins) |
| 13×13, 1024ch, 7×7 | (native) | **7** | 24 | 22 | 1.0× (native) |

### 3.2 End-to-End Model Latency (model.half(), channels_last, B=1) — ✅

**ConvMixer-1536/20:** cuDNN NCHW eager = 3.56 ms (best). Custom + compile(r-o) = **1.89 ms (1.88×)**.
**ConvMixer-768/32:** cuDNN NHWC + compile(r-o) = 1.14 ms (best). Custom + compile(r-o) = **0.87 ms (1.31×)**.
**RepLKNet-31B:** cuDNN NCHW eager = 5.09 ms (best). Custom + compile(r-o) = **1.62 ms (3.14×)**.

### 3.3 Baseline Comparison — ✅

ConvMixer-768/32 (0.87 ms) is 1.48× faster than MobileNetV3-Large (1.29 ms) with 5.0% higher accuracy.

### 3.4 NCU Profiling — ✅ (15/17 profiles)

Custom L2 hit: 76-96% vs cuDNN NCHW: 42-77% on Ada. Full metrics in `results/ada_ncu_metrics.csv`.

### 3.5 Compile-Survival — ✅

cuDNN NCHW: 167.7 → 1543.2 μs (9.2× degradation). Custom NHWC: 163.7 → 126.0 μs (1.30× faster). Platform-independent.

### 3.6 Tile Sensitivity (31×31/128ch, M sweep) — ✅

M=4 fastest (0.249 ms, 5.5 blocks/SM). M=16 catastrophic (0.681 ms, 0.45 blocks/SM). Output: `results/ada_sensitivity_31x31.csv`.

### 3.7 Batch Sweep — Custom and cuDNN-NHWC only — ⚠ PARTIAL (see §4)

Prior batch sweep only measured Custom and cuDNN-**NHWC**. cuDNN's best layout (NCHW) and Triton were **not** included. This is a methodology gap — §4 specifies the fix.

---

## 4. NEW EXPERIMENTS — Batch Sweep with All Methods, Correct Layouts

### Why These Are Needed (C0 from DS_suggestion.md)

The paper's stated methodology (§5.1) is: *"We report the best cuDNN configuration as the baseline."* The B=1 E2E tables follow this correctly (cuDNN-NCHW = 3.56 ms at B=1 for 1536/20). But the batch sweep in §5.6 silently uses cuDNN-NHWC (135 img/s at B=1 vs 281 img/s for NCHW).

**A reviewer WILL notice this inconsistency.** The batch sweep is the paper's most important new experiment — it is the evidence that distinguishes "characterization" from "engineering." If cuDNN's worst layout was used, the BW-saturation claim is unverified.

Additionally, Triton has **zero** batch scaling data. If Triton scales better with batch than Custom, the "Triton can't express cross-program shared-memory reuse" claim only holds at B=1.

### 4.1 Measure cuDNN-NCHW Batch Scaling (~15 min)

ConvMixer-1536/20, model.half(), real ImageNet, mode=default compile:

```bash
export CUDA_VISIBLE_DEVICES=1 && conda activate cuda
cd /home/chenchon/chConv/paper/latex/repo

# Need to modify convmixer/benchmark.py to support --cudnn_layout flag, OR
# manually construct the cuDNN-NCHW model and run the batch throughput function.

# Approach: use benchmark_throughput_real() with a manually-built NCHW model.
# The cuDNN NCHW model is: load model, replace_depthwise_convs (to get custom),
# but instead keep the original model in NCHW (NOT channels_last).

# Quick method: modify the benchmark to accept --layout nchw, or
# write a short script that:
# 1. loads model without channels_last conversion
# 2. converts to NCHW explicitly
# 3. runs get_imagenet_dataloader + benchmark_throughput_real

python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 1 --real_data \
    --data_dir /home/chenchon/chConv/data/imagenet --cudnn_layout nchw \
    2>&1 | tee results/ada_batch_cudnn_nchw_b1.log

python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 4 --real_data \
    --data_dir /home/chenchon/chConv/data/imagenet --cudnn_layout nchw \
    2>&1 | tee results/ada_batch_cudnn_nchw_b4.log

python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 8 --real_data \
    --data_dir /home/chenchon/chConv/data/imagenet --cudnn_layout nchw \
    2>&1 | tee results/ada_batch_cudnn_nchw_b8.log
```

**If `--cudnn_layout nchw` flag doesn't exist yet**: Add it to `convmixer/benchmark.py`. The flag should:
- Skip calling `model.to(memory_format=torch.channels_last)` 
- Load the model in NCHW (default contiguous format)
- Pass NCHW tensors to the model

Record: latency (ms) and throughput (img/s) at each batch size.

### 4.2 Measure Triton Batch Scaling (~15 min)

ConvMixer-1536/20 with Triton depthwise kernels, model.half(), real ImageNet:

```bash
# Need a variant of the model where depthwise layers are replaced with Triton ops.
# If the Triton integration exists as a model variant or flag:
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 1 --real_data \
    --data_dir /home/chenchon/chConv/data/imagenet --triton_mode \
    2>&1 | tee results/ada_batch_triton_b1.log

python convmixer/benchmark.py --model 1536_20 --skip_accuracy --num_runs 200 \
    --device cuda:0 --fp16_method model_half --batch 4 --real_data \
    --data_dir /home/chenchon/chConv/data/imagenet --triton_mode \
    2>&1 | tee results/ada_batch_triton_b4.log
```

**If `--triton_mode` doesn't exist**: The simplest approach is to write a 20-line script that:
1. Loads ConvMixer-1536/20
2. Replaces each depthwise layer with a wrapper calling the Triton kernel
3. Runs the benchmark_throughput_real loop

Record: latency and throughput at B=1, B=4.

### 4.3 Expected Findings

| Scenario | Interpretation |
|----------|---------------|
| cuDNN-NCHW scales similarly to NHWC (flat/declining) | BW-saturation claim confirmed — cuDNN doesn't benefit from batch regardless of layout |
| cuDNN-NCHW scales BETTER than NHWC with batch | BW-saturation claim weakens — cuDNN's best layout mitigates the bottleneck. Paper must acknowledge this |
| Triton throughput flat/declining with batch | "Triton can't express cross-program shmem reuse" holds at all batch sizes |
| Triton throughput IMPROVES with batch | Paper must acknowledge Triton recovers at batch>1 through batch-level parallelism |

### 4.4 After Measuring

Update `revision/results.md` with the new batch table. Update paper.tex Table `tab:batch` to include cuDNN-NCHW and Triton columns. Reconcile any narrative changes (especially if cuDNN-NCHW scales better than expected).

---

## 5. Remaining Paper Analysis (No GPU Needed)

### 5.1 Install matplotlib + roofline plot (~15 min)
```bash
pip install matplotlib
python plot_roofline.py --ncu_csv results/ada_ncu_metrics.csv \
    --output results/ada_roofline_plot.png \
    --fp16_peak_tflops 91.1 --bandwidth_gbps 960
```

### 5.2 Extract L2 hit rates from NCU CSVs (~15 min)
```bash
python analyze_ncu.py --ncu_dir results/ncu/ --output results/ada_ncu_l2_metrics.csv --metric l2
```

---

## 6. Paper Text Update Status

| Section | Status |
|---------|--------|
| Abstract | ✅ |
| §4 (PyTorch Integration) | ✅ |
| §5.1 (Methodology) | ✅ |
| §5.3 (E2E tables) | ✅ |
| §5.5 (Baselines) | ✅ |
| §7.3 (Implications) | ✅ |
| §7.4 (System-Level) | ✅ |
| §8 (Conclusion) | ✅ |
| **§2.7 (L2 residency analysis)** | **⬜** |
| **§5.2 (Per-kernel 6 configs × 3 methods)** | **⬜** |
| **§5.4 (NCU L2 analysis)** | **⬜** |
| **§5.6 (Batch scaling — add cuDNN-NCHW + Triton columns)** | **⬜ BLOCKED on §4** |
| **§5.7 (Tile sensitivity — add Spark column)** | **⬜** |
| **§7.1 (BW-pressure principle)** | **⬜ — replace "Why Shared Memory Works"** |

---

## 7. Output Files

```
results/
├── ada_cudnn_layouts.csv                    ✅
├── ada_triton_results.csv                   ✅
├── ada_ncu_metrics.csv                      ✅
├── ada_ncu_l2_metrics.csv                   ⬜ §5.2
├── ada_roofline_plot.png                    ⬜ §5.1
├── ada_nsys_compile_fusion.log              ✅
├── ada_1536_20_model_half.log               ✅
├── ada_768_32_model_half_b1.log             ✅
├── ada_replknet_model_half_b1.log           ✅
├── ada_baselines_model_half.log             ✅
├── ada_sensitivity_31x31.csv                ✅
├── ada_batch_cudnn_nchw_b1.log              ⬜ §4.1 — NEW
├── ada_batch_cudnn_nchw_b4.log              ⬜ §4.1 — NEW
├── ada_batch_cudnn_nchw_b8.log              ⬜ §4.1 — NEW
├── ada_batch_triton_b1.log                  ⬜ §4.2 — NEW
├── ada_batch_triton_b4.log                  ⬜ §4.2 — NEW
└── ncu/                                     ✅ (15/17 profiles)
```

---

## 8. Known Pitfalls

1. **Stale ninja lock**: `rm -rf ~/.cache/torch_extensions/py310_cu128/depthwise_*`
2. **torch.compile reduce-overhead on RepLKNet**: First compile 10-20 min. Cached in `/tmp/torchinductor_chenchon/`.
3. **GPU 0 occupied**: Always `CUDA_VISIBLE_DEVICES=1`.
4. **v2 kernel**: Documented dead-end behind `DEPTHWISE_9X9_VARIANT=v2` env toggle. Do not promote.
5. **cuDNN-NCHW batch testing**: Need to ensure the model runs in pure NCHW (no channels_last conversion). The `--cudnn_layout nchw` flag may need to be added to `convmixer/benchmark.py`.
6. **Triton batch testing**: The Triton integration for E2E models (not just per-kernel) may need a script. Check `benchmark_triton.py` for the E2E integration pattern.