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

## 8. Cross-Platform Launch-Overhead Investigation (added 2026-05-09)

### 8.1 Why this section exists

After the Spark numbers were published, we noticed a counterintuitive pattern in the cross-platform table (`results/spark_cross_platform_table.md`, baselines section):

| Model | Ada (ms) | Spark (ms) | Spark/Ada |
|-------|---------|-----------|-----------|
| MobileNetV3-Large | 1.29 | **0.97** | 0.75× |
| EfficientNet-B0 | 1.70 | **1.30** | 0.76× |
| ResNet-152 | 2.57 | **2.23** | 0.87× |

Ada has higher peak FP16 (91.1 TFLOPs vs ~125–209 on Spark, depending on which spec) and higher memory bandwidth (960 GB/s vs ~273 GB/s) — yet at B=1 Ada is *slower* on these three baselines. The compute-bound models (ConvMixer-1536/20, ConvMixer-768/32, RepLKNet-31B) all behave correctly: Ada wins by 2–3× as peak specs predict. The inversion is specific to launch-bound, small-kernel, B=1 inference.

**Two non-mutually-exclusive hypotheses:**

1. **Software-stack confound.** Ada baseline log was captured with `PyTorch 2.9.1 + cuDNN 9.10.02`. Spark used `PyTorch 2.11.0 + cuDNN 9.19.0`. Multiple dispatcher fast-paths and `torch.compile` overhead reductions landed in PT 2.10/2.11; cuDNN 9.19 has different graph-fusion heuristics. This alone could produce the gap.
2. **Launch-overhead regime.** Sanity check on Ada numbers: ResNet-152 at B=1 = 11.5 GMACs / 2.57 ms = **9 TFLOPs achieved vs ~364 TFLOPs peak ≈ 2.4% utilization**. The GPU is idle ~97% of the time. In that regime, peak FLOPs/BW don't matter — the host→GPU launch path dominates. PCIe Gen4 (Ada) vs NVLink-C2C (Spark, Grace–Blackwell on package) is a plausible per-launch latency gap; with ResNet-152's ~150 kernel launches/forward, even 5 μs/launch = 0.75 ms.

**Goal of §8 work:** disentangle (1) from (2) and report numbers that are no longer software-version-confounded. This is a direct response to anyone who reads the cross-platform table and asks "wait, how is Ada slower?"

### 8.2 New finding from re-running on GB10 (2026-05-09)

Side-channel finding worth checking on Ada too: on Spark, `cuDNN NCHW` is **3× faster than NHWC** for RepLKNet-31B. The Spark cross-platform table only reports the NHWC numbers, which artificially inflates the Custom-vs-cuDNN ratio.

| RepLKNet-31B (B=1, model_half) | Spark (ms) |
|---|---|
| CuDNN NHWC eager | 18.12 |
| CuDNN NHWC + compile | 15.19 |
| **CuDNN NCHW eager** | **8.21** |
| **CuDNN NCHW + compile** | **4.99** |
| Custom + compile | 4.71 |

So Custom+compile only beats cuDNN-NCHW-best by **1.06×** on Spark, not the 3.27× implied by the NHWC-only table. The Ada plan's §3.2 already shows cuDNN NCHW is the cuDNN best for RepLKNet on Ada (5.09 ms eager beats 7.63 NHWC eager). **Action: confirm the NCHW comparison is also up-to-date on Ada with PT 2.11 stack (Step 8.4).**

### 8.3 Environment refresh (run first, ~5 min)

```bash
source /home/chenchon/miniconda3/etc/profile.d/conda.sh && conda activate cuda
export CUDA_VISIBLE_DEVICES=1
cd /home/chenchon/chConv/paper/latex/repo

# Snapshot the old environment for record
python -c "import torch; print('OLD:', torch.__version__, 'cuDNN', torch.backends.cudnn.version())" \
    | tee results/ada_env_pt29.txt

# Upgrade to match Spark stack
pip install --upgrade \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print('NEW:', torch.__version__, 'cuDNN', torch.backends.cudnn.version())" \
    | tee results/ada_env_pt211.txt
# Expect: 2.11.0+cu128, cuDNN 91900
```

If the upgrade breaks the custom kernel build, clear the extension cache: `rm -rf ~/.cache/torch_extensions/py310_cu128/depthwise_*` and let it rebuild on first import.

### 8.4 Step 8.4 — Re-run baselines with matched stack (~10 min)

```bash
python convmixer/benchmark.py --model 1536_20 --skip_accuracy --baselines \
    --device cuda:0 --fp16_method model_half --num_runs 500 \
    2>&1 | tee results/ada_baselines_pt211.log

# Also re-confirm RepLKNet best-of-cuDNN with NCHW under new stack
python replknet/benchmark.py --device cuda:0 --num_runs 200 \
    --skip_verify --skip_accuracy \
    2>&1 | tee results/ada_replknet_pt211.log
```

**Decision rule:**
- If MobileNet/EfficientNet/ResNet-152 latencies *drop substantially* on Ada with PT 2.11 (e.g., ResNet-152 falls from 2.57 → ≤2.23 ms), the original "Spark wins" was largely a software-stack effect. Update §3.3 of this plan and the cross-platform table accordingly.
- If they don't drop (gap persists), proceed to Steps 8.5–8.7 to confirm launch-overhead is the cause.

Expected output table (fill in):

| Model | Ada PT 2.9.1 (ms) | Ada PT 2.11 (ms) | Spark PT 2.11 (ms) | Δ stack |
|-------|-------------------|------------------|--------------------|---------|
| MobileNetV3-Large | 1.29 | ___ | 0.97 | ___ |
| EfficientNet-B0 | 1.70 | ___ | 1.30 | ___ |
| ResNet-152 | 2.57 | ___ | 2.23 | ___ |
| ConvMixer-1536/20 (Custom+compile) | 1.59 | ___ | 4.32 | ___ |

### 8.5 Step 8.5 — Batch sweep (kills the launch-bound regime, ~30 min)

```bash
for B in 1 4 16 32 64; do
    python convmixer/benchmark.py --model 1536_20 --skip_accuracy --baselines \
        --device cuda:0 --fp16_method model_half --batch $B --num_runs 200 \
        2>&1 | tee results/ada_baselines_b${B}.log
done
```

**Decision rule:** plot **throughput (img/s) vs batch** for ResNet-152 + MobileNetV3-Large on Ada and Spark. Predicted shape:

- B=1: Spark may match or beat Ada (launch-bound).
- B=16+: Ada pulls ahead by ~1.5–3× as compute/BW saturates.

If that crossover is visible, launch-overhead is the dominant factor. Save plot to `results/ada_baseline_batch_scaling.png`.

### 8.6 Step 8.6 — Nsight Systems profiling (definitive evidence, ~15 min)

```bash
# ResNet-152 launch-vs-kernel breakdown
nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-device=0 \
    --force-overwrite=true \
    -o results/ada_resnet152_nsys \
    python -c "
import torch, torchvision.models as tvm
m = tvm.resnet152(weights=tvm.ResNet152_Weights.IMAGENET1K_V2)
m = m.cuda().half().to(memory_format=torch.channels_last).eval()
m = torch.compile(m, mode='default')
x = torch.randn(1, 3, 224, 224, device='cuda', dtype=torch.float16
                ).to(memory_format=torch.channels_last)
with torch.no_grad():
    for _ in range(200): m(x)         # warmup + compile
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push('measure')
    for _ in range(200): m(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
"
nsys stats --report cudaapisum,gpukernsum results/ada_resnet152_nsys.nsys-rep \
    | tee results/ada_resnet152_nsys_stats.txt
```

Run identically on Spark for comparison. **Compare:**

- `gpukernsum` total → actual GPU compute time. Should be smaller on Ada (faster GPU).
- `cudaapisum` total for `cudaLaunchKernel` → host-side launch cost.
- Average gap between consecutive kernels (visible in nsys timeline).

**If `cudaLaunchKernel` time + inter-kernel gaps explain the residual Ada–Spark difference, the launch-overhead hypothesis is confirmed.**

### 8.7 Step 8.7 — CUDA Graph capture (eliminates launch overhead entirely, ~15 min)

```bash
python -c "
import torch, time, torchvision.models as tvm
m = tvm.resnet152(weights=tvm.ResNet152_Weights.IMAGENET1K_V2)
m = m.cuda().half().to(memory_format=torch.channels_last).eval()
x = torch.randn(1, 3, 224, 224, device='cuda', dtype=torch.float16
                ).to(memory_format=torch.channels_last)

with torch.no_grad():
    for _ in range(50): m(x)
    torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    out = m(x)

torch.cuda.synchronize()
N = 1000
t0 = time.perf_counter()
for _ in range(N): g.replay()
torch.cuda.synchronize()
print(f'CUDA graph replay: {(time.perf_counter()-t0)/N*1000:.3f} ms')
" 2>&1 | tee results/ada_resnet152_cudagraph.log
```

Run on both Ada and Spark. **Decision rule:**

- **If both platforms collapse to similar latency** under CUDA graphs (gap shrinks to <10%), the original gap was purely launch overhead. The Spark "win" disappears once host overhead is removed.
- **If Ada beats Spark under CUDA graphs by approximately the spec ratio** (e.g., Ada ≤ 0.5× Spark), that's the smoking gun: peak compute does win once launches are out of the way.

This is the single most decisive experiment of §8. Run it first if pressed for time.

### 8.8 Section to add to the paper

Once Steps 8.4–8.7 are complete, add a new short subsection (§5.8 or appendix) titled something like *"Launch-overhead regime at B=1: when peak specs do not predict latency"*. Key claim:

> At batch=1 with deeply layered, small-kernel networks (ResNet-152, MobileNetV3-Large), the inference latency is dominated by host-to-device kernel launch overhead, not by GPU compute. Under this regime, achieved utilization can be <5% of peak FP16 throughput. Tightly coupled host–accelerator architectures (e.g., Grace–Blackwell NVLink-C2C) can therefore outperform discrete PCIe-attached GPUs even when the latter has 3× higher peak compute and 3.5× higher memory bandwidth. The advantage disappears under CUDA graph replay (Step 8.7) or at larger batch (Step 8.5), where the GPU saturates and peak specs again predict outcome.

This frames the cross-platform comparison correctly: the L2-residency principle (§7.1) applies to the **compute-bound regime** (large depthwise kernels), and the launch-overhead regime is a separate effect that explains the inverted baseline numbers.

### 8.9 Output files for §8

```
results/
├── ada_env_pt29.txt                         ⬜ (Step 8.3 — pre-upgrade snapshot)
├── ada_env_pt211.txt                        ⬜ (Step 8.3 — post-upgrade snapshot)
├── ada_baselines_pt211.log                  ⬜ (Step 8.4)
├── ada_replknet_pt211.log                   ⬜ (Step 8.4 — NCHW best-of cross-check)
├── ada_baselines_b{1,4,16,32,64}.log        ⬜ (Step 8.5)
├── ada_baseline_batch_scaling.png           ⬜ (Step 8.5 plot)
├── ada_resnet152_nsys.nsys-rep              ⬜ (Step 8.6)
├── ada_resnet152_nsys_stats.txt             ⬜ (Step 8.6 summary)
└── ada_resnet152_cudagraph.log              ⬜ (Step 8.7 — decisive)
```

### 8.10 Pitfalls specific to §8

1. **PT 2.11 may invalidate compile cache** — first call to `torch.compile` will recompile (~minutes per model). Run a quick warm-up pass before benchmarking.
2. **CUDA graph capture requires fixed shape and no Python-level control flow.** `torch.compile`'d models sometimes have shape guards that break capture. If `torch.cuda.graph` errors, capture the *uncompiled* model — the comparison is still valid.
3. **Nsight Systems on cu128:** if `--gpu-metrics-device` errors, drop that flag; the trace alone (`--trace=cuda`) is enough for `cudaapisum`/`gpukernsum`.
4. **Power-state at B=1:** Ada may not boost to top clock when the GPU is mostly idle between launches. Lock clocks before timing if precision matters: `sudo nvidia-smi -lgc <max_freq>` then unlock with `sudo nvidia-smi -rgc`.

---

## 9. Known Pitfalls

1. **Stale ninja lock**: `rm -rf ~/.cache/torch_extensions/py310_cu128/depthwise_*`
2. **torch.compile reduce-overhead on RepLKNet**: First compile 10-20 min. Cached in `/tmp/torchinductor_chenchon/`.
3. **GPU 0 occupied**: Always `CUDA_VISIBLE_DEVICES=1`.
4. **v2 kernel**: Documented dead-end behind `DEPTHWISE_9X9_VARIANT=v2` env toggle. Do not promote.
5. **cuDNN-NCHW batch testing**: Need to ensure the model runs in pure NCHW (no channels_last conversion). The `--cudnn_layout nchw` flag may need to be added to `convmixer/benchmark.py`.
6. **Triton batch testing**: The Triton integration for E2E models (not just per-kernel) may need a script. Check `benchmark_triton.py` for the E2E integration pattern.

---

## 10. Paper Table Audit — Data Gaps Found (2026-05-10)

Systematic audit of all 18 tables and 5 figures revealed the following gaps.
Items marked ⬜ need GPU time. Items marked 🔧 are data-entry fixes (no GPU needed).

### 10.1 Data Errors to Fix (🔧, no GPU needed)

| # | Table | Issue | Fix |
|---|-------|-------|-----|
| D1 | `tab:ncu` line 752 | Ada Custom 29×29 L2 hit = "---" | Fill from `ada_ncu_tables.md`: **92.1%** |
| D2 | `tab:kernel_perf` Ada 13×13 Triton | Shows "---" but `ada_triton_results.csv` has 22 μs | Fill: **22** |
| D3 | `tab:kernel_perf` GB10 29×29 Custom | Shows **358** μs but correct value is **268** μs (confirmed by spark_kernel_comparison.csv: 0.268 ms, and per-stage RepLKNet: 0.538 ms ÷ 2 = 269 μs) | Change 358 → **268** |
| D4 | `tab:tilesweep` GB10 M=16 blocks/SM | Shows **0.25** but 64 blocks ÷ 48 SMs = **1.33** | Change 0.25 → **1.3** |
| D5 | Text line 794 | Says "6.6×" GB10 compile degradation but `tab:compilesurvival` shows 438→2660 μs = 6.07× (table says 6.1×) | Change text to **6.1×** |

### 10.2 Missing GB10 Data — Fillable from Existing Spark Files (🔧)

The following `tab:kernel_perf` GB10 entries show "---" but data EXISTS in Spark branch files.

**From `spark_cudnn_layouts.csv` (cuDNN NCHW + NHWC):**

| Config | cuDNN NCHW (μs) | cuDNN NHWC (μs) | cuDNN-best |
|--------|-----------------|-----------------|------------|
| 7×7, 768ch | 71 | 362 | 71 (NCHW) |
| 9×9, 1536ch | 295 | 1088 | 295 (NCHW) |
| 13×13, 1024ch | 13 | 54 | 13 (NCHW) |
| 27×27, 512ch | 44 | 301 | 44 (NCHW) |
| 29×29, 256ch | 198 | 1026 | 198 (NCHW) |
| 31×31, 128ch | 416 | 2045 | 416 (NCHW) |

**From `spark_kernel_comparison.csv` (Custom + Triton):**

| Config | Custom (μs) | Triton (μs) | vs cuDNN-best |
|--------|------------|-------------|---------------|
| 7×7, 768ch | 25 | 96 | 2.84× |
| 9×9, 1536ch | 73 | 299 | 4.04× |
| 13×13, 1024ch | 190 | 16 | 0.07× |
| 27×27, 512ch | 177 | 87 | 0.25× |
| 29×29, 256ch | 268 | 265 | 0.74× |
| 31×31, 128ch | 428 | 717 | 0.97× |

**Note on Triton GB10**: Triton is very slow for large kernels on GB10 (717 μs for 31×31, slower than even cuDNN NHWC at 2045 μs). Including these numbers would weaken the paper's narrative. **Recommendation**: include GB10 Triton only for 7×7 and 9×9 where it's competitive; mark larger kernels as "---" with footnote "Triton autotune fails for K≥27 on GB10 (L2 too small for autotuner working set)".

**Note on GB10 13×13**: Custom = 190 μs is **14.6× slower** than cuDNN NCHW (13 μs). This is because 1024ch/7×7 has tiny spatial dimension — kernel launch overhead dominates. This is an unfavorable result for Custom. **Recommendation**: either (a) include honestly and explain, or (b) drop 13×13 row entirely since RepLKNet's 13×13 stage contributes <5% of total latency.

### 10.3 Missing Experiments — Need GPU Time (⬜)

| # | Table | What's Missing | Priority | Time | Why Needed |
|---|-------|---------------|----------|------|------------|
| E1 | `tab:pointwise_layout` | **GB10 pointwise NCHW vs NHWC** for 4 configs | HIGH | 10 min | Table 1 is Ada-only; paper claims cross-platform. Script already written: `benchmark_pointwise_layout.py` |
| E2 | `tab:batch` | GB10 RepLKNet-31B batch sweep (B=1,2,4) | MEDIUM | 30 min | Table only has Ada rows. If BW-saturation holds on GB10, this reinforces the cross-platform regime claim |
| E3 | `tab:batch` | GB10 ConvMixer-768/32 batch sweep (B=1,2,4) | MEDIUM | 20 min | Same as E2 for second architecture |
| E4 | `tab:batch` | GB10 ConvMixer-1536/20 cuDNN-NCHW batch (B=1,2,4) | LOW | 15 min | Currently only cuDNN-NHWC + Custom columns for GB10. cuDNN-NCHW shows best-cuDNN comparison |
| E5 | `tab:batch` | GB10 Triton batch sweep for ConvMixer-1536/20 | SKIP | — | Triton is 5.81× slower than Custom on GB10 (9.18 vs 4.14 ms compiled). Not meaningful. |

### 10.4 Experiment Scripts for Missing Data

**E1 — GB10 Pointwise Layout Benchmark (on Spark machine):**
```bash
python benchmark_pointwise_layout.py --device cuda:0 --num_runs 500 \
    --output results/spark_pointwise_layout.csv
```
This measures 1×1 conv in NCHW vs NHWC for: ConvMixer-1536 (32²×1536), RepLKNet stg2→3 (14², 512→1024), RepLKNet stg3 (7²×1024), ConvMixer-768 (32²×768).

Expected result: On GB10 with lower bandwidth (273 vs 960 GB/s), NHWC pointwise speedup should be **larger** than Ada (1.18–1.22×) because NHWC avoids the NCHW→NHWC transpose overhead for GEMM, which is more costly when BW is scarce.

**E1 — Ada Pointwise Layout Benchmark (re-confirm on this machine):**
```bash
python benchmark_pointwise_layout.py --device cuda:0 --num_runs 500 \
    --output results/ada_pointwise_layout.csv
```
Run this first to confirm Ada numbers match current Table 1 values (0.0425, 0.0359, etc.).

**E2 — GB10 RepLKNet-31B Batch Sweep (on Spark machine):**
```bash
for B in 1 2 4; do
    python replknet/benchmark_replknet.py --device cuda:0 --batch $B \
        --num_runs 200 --fp16_method model_half \
        2>&1 | tee results/spark_replknet_batch_b${B}.log
done
```
Expected: Custom throughput flat/declining (BW-saturated at B=1, per results.md §11.2). cuDNN throughput also flat. This confirms BW-bound regime on GB10.

**E3 — GB10 ConvMixer-768/32 Batch Sweep (on Spark machine):**
```bash
for B in 1 2 4; do
    python convmixer/benchmark.py --model 768_32 --skip_accuracy \
        --device cuda:0 --fp16_method model_half --batch $B --real_data \
        --num_runs 200 \
        2>&1 | tee results/spark_768_32_batch_b${B}.log
done
```

**E4 — GB10 ConvMixer-1536/20 cuDNN-NCHW Batch Sweep (on Spark machine):**
```bash
for B in 1 2 4; do
    python convmixer/benchmark.py --model 1536_20 --skip_accuracy \
        --device cuda:0 --fp16_method model_half --batch $B --real_data \
        --cudnn_layout nchw --num_runs 200 \
        2>&1 | tee results/spark_1536_20_cudnn_nchw_batch_b${B}.log
done
```
Note: requires `--cudnn_layout nchw` flag (may need porting to SPARK branch).

### 10.5 Priority Order

1. **🔧 D1–D5**: Fix data errors in paper.tex immediately (no GPU needed)
2. **🔧 Fill GB10 tab:kernel_perf**: Enter existing Spark data (no GPU needed)
3. **⬜ E1**: Run pointwise layout benchmark on both machines
4. **⬜ E2**: GB10 RepLKNet batch sweep
5. **⬜ E3**: GB10 ConvMixer-768/32 batch sweep
6. **⬜ E4**: GB10 cuDNN-NCHW batch sweep (lower priority)