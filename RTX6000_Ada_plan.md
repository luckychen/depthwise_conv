# RTX 6000 Ada Experiment Plan — Cross-Platform Pair for Revision

This file is the executable instruction sheet for the RTX 6000 Ada workstation. It pairs with the DGX Spark experiments (`SPARK` branch, `SPARK_context.md`) so the revised paper presents a two-platform L2 residency regime study.

**Read this entire file before doing anything.** Follow the numbered action items exactly.

## 0. Why Ada Matters for the Revision

Reviewer R3 said "single GPU" was a reject-grade flaw. Reviewer R4 said "no conceptual insight." The revision answers both with an **L2 residency principle**: shared-memory weight tiling matters when (working set) > (L2 capacity). Ada (SM 8.9, 142 SMs, **96 MB L2**, 960 GB/s, 91.1 FP16 TFLOPS) and Spark (SM 12.1, 48 SMs, **~24 MB unified**, ~400 GB/s) are deliberately chosen for their 4.4× L2 capacity difference. The expected finding: on Ada, the large L2 absorbs the depthwise working set for most configs, so custom shmem tiling provides diminishing returns relative to Spark. This is the paper's central causal claim.

**Status (2026-05-08)**:
- All experiments: **COMPLETE**. No new Ada experiments needed.
- Paper analysis: L2 hit rate extraction from existing NCU data (Step 5) remains.
- Roofline plot: blocked on matplotlib (Step 5.1).

## 1. Hardware Specs

| Property | Ada (this machine) | Spark |
|---|---|---|
| GPU | NVIDIA RTX 6000 Ada | DGX Spark (GB10) |
| Architecture | Ada Lovelace (SM 8.9) | Blackwell (SM 12.1) |
| SM Count | 142 | 48 |
| L2 Cache | **96 MB** | **~24 MB** (4.4× smaller) |
| Memory | 48 GB GDDR6 | 128 GB unified |
| Memory Bandwidth | 960 GB/s | ~400 GB/s (2.4× lower) |
| FP16 Peak | 91.1 TFLOPS | ~209 TFLOPS |

## 2. Environment

```bash
source /home/chenchon/miniconda3/etc/profile.d/conda.sh && conda activate cuda
export CUDA_VISIBLE_DEVICES=1
cd /home/chenchon/chConv/paper/latex/repo
```

Stack: Python 3.10, PyTorch 2.9.1+cu128, CUDA 12.8 (nvcc 12.9), cuDNN 9.10.2.

### FP16 Methodology

**Use `model.half()` as the primary methodology.** All revision numbers use this. Autocast kept as secondary column for paper continuity.

| Method | How | When |
|--------|-----|------|
| **model.half()** (primary) | `model = model.half()` then `model(x.half())` | All revision numbers |
| autocast (secondary) | FP32 model, `torch.amp.autocast` per forward | Paper continuity only |

## 3. Experiment Results (All Complete)

### 3.1 Per-Kernel Depthwise — ✅

| Config | Custom (μs) | cuDNN NCHW (μs) | cuDNN NHWC (μs) | Triton (μs) | Custom vs best cuDNN |
|--------|------------|-----------------|-----------------|-------------|---------------------|
| 9×9, 1536ch, 32×32 | 28.1 | 90 | 293 | 127 | **3.2×** (vs NCHW) |
| 7×7, 768ch, 32×32 | 27.7 | 27 | **18** | 43 | 0.65× (cuDNN NHWC wins) |
| 31×31, 128ch, 56×56 | 116 | **150** | 634 | 286 | 1.3× |
| 29×29, 256ch, 28×28 | 72 | **77** | 267 | 128 | 1.07× |
| 27×27, 512ch, 14×14 | 22 | **15** | 93 | 38 | 0.68× (cuDNN NCHW wins) |
| 13×13, 1024ch, 7×7 | (native) | **7** | 24 | 22 | 1.0× (native) |

**L2 interpretation**: cuDNN NHWC only wins for 7×7 (specialized kernel exists). For 27×27/512, cuDNN NCHW wins per-kernel (better coalescing on small spatial), but this reverses at network level (see §3.3).

### 3.2 End-to-End Model Latency (model.half(), channels_last, B=1) — ✅

**ConvMixer-1536/20 (9×9, 51.6M params):**

| Configuration | Latency | vs CuDNN best |
|---------------|---------|---------------|
| cuDNN NCHW eager | **3.56 ms** | 1.00× (best cuDNN) |
| cuDNN NHWC eager | 8.84 ms | — |
| cuDNN NHWC +compile(default) | 8.51 ms | — |
| Custom NHWC eager | 2.25 ms | 1.58× |
| Custom NHWC +compile(default) | 1.90 ms | 1.87× |
| **Custom NHWC +compile(r-o)** | **1.89 ms** | **1.88×** |

**ConvMixer-768/32 (7×7, 21.1M params):**

| Configuration | Latency | vs CuDNN best |
|---------------|---------|---------------|
| cuDNN NHWC +compile(r-o) | **1.14 ms** | 1.00× (best cuDNN) |
| cuDNN NHWC eager | 2.52 ms | — |
| cuDNN NCHW eager | 2.25 ms | — |
| Custom NHWC eager | 2.68 ms | 0.43× |
| Custom NHWC +compile(default) | 1.26 ms | 0.90× |
| **Custom NHWC +compile(r-o)** | **0.87 ms** | **1.31×** |

**Key**: 7×7 is the L2 regime boundary. cuDNN has a specialized NHWC kernel here. Custom eager loses (dispatch overhead across 32 layers). But compile eliminates dispatch + CUDA graphs eliminate launch overhead → Custom pulls ahead. On Spark (small L2), Custom wins even at the per-kernel level (25 vs 71 μs).

**RepLKNet-31B (27-31×31, 79M params):**

| Configuration | Latency | vs CuDNN best |
|---------------|---------|---------------|
| cuDNN NCHW eager | **5.09 ms** | 1.00× (best cuDNN) |
| cuDNN NCHW +compile(default) | 6.85 ms | 0.74× (compile hurts!) |
| cuDNN NHWC eager | 7.63 ms | — |
| Custom NHWC eager | 5.18 ms | 0.98× |
| Custom NHWC +compile(default) | 2.51 ms | 2.03× |
| **Custom NHWC +compile(r-o)** | **1.62 ms** | **3.14×** |

### 3.3 Baseline Model Comparison (model.half(), torch.compile reduce-overhead) — ✅

| Model | Params | Top-1 | Latency | Acc/ms |
|-------|--------|-------|---------|--------|
| MobileNetV3-Large | 5.5M | 75.15% | 1.29 ms | 58.3 |
| **ConvMixer-768/32 (custom+r-o)** | **21M** | **80.15%** | **0.87 ms** | **92.1** |
| **RepLKNet-31B (custom+r-o)** | **80M** | **83.30%** | **1.62 ms** | **51.4** |
| **ConvMixer-1536/20 (custom+r-o)** | **52M** | **81.36%** | **1.89 ms** | **43.1** |
| EfficientNet-B0 | 5.3M | 77.68% | 1.70 ms | 45.7 |
| ResNet-152 | 60M | 82.19% | 2.57 ms | 32.0 |

### 3.4 NCU Profiling — ✅ (15/17 profiles)

Full metrics in `results.md` §9. Key L2-relevant findings:

| Config | Method | GFLOP/s | Mem BW (%) | Occupancy | L1 Hit (%) | L2 Hit (%) |
|--------|--------|---------|-----------|-----------|-----------|-----------|
| 9×9, 1536ch | Custom | 3628 | 70.1 | 55.1 | 21.7 | 81.7 |
| 9×9, 1536ch | cuDNN NCHW | 1066 | 80.1 | 62.2 | 99.2 | 51.6 |
| 7×7, 768ch | Custom | 2734 | 51.6 | 63.1 | 90.8 | 75.9 |
| 7×7, 768ch | cuDNN NCHW | 1133 | 52.7 | 63.2 | 97.9 | 54.9 |
| 31×31, 128ch | Custom | 2742 | 59.7 | 32.4 | 89.4 | 95.9 |
| 31×31, 128ch | cuDNN NCHW | 1889 | 78.7 | 61.9 | 99.8 | 76.5 |
| 27×27, 512ch | Custom | 3820 | 33.2 | 30.2 | 73.1 | 93.0 |
| 27×27, 512ch | cuDNN NCHW | 3794 | 52.9 | 63.0 | 99.0 | 41.6 |

**L2 finding**: Custom kernels achieve consistently higher L2 hit rates (76-96%) than cuDNN NCHW (42-77%), especially for large-K configs. Ada's 96MB L2 is large enough that Custom's working set stays mostly resident. On Spark (~24MB), the L2 hit rate gap should be even larger — explaining why Custom's relative advantage grows on small-L2 platforms.

### 3.5 Compile-Survival — ✅

| Method | Eager (μs) | Compiled (μs) | Result |
|--------|-----------|---------------|--------|
| Custom NHWC | 163.7 | 126.0 | **1.30× faster** |
| cuDNN NCHW | 167.7 | **1543.2** | **9.2× degradation** |

**Platform-independent finding.** Confirmed on both Ada (9.2×) and Spark (6.6×).

### 3.6 Batch Sweep (ConvMixer-1536/20, model.half()) — ✅

| B | Custom eager (img/s) | Custom+compile (img/s) | CuDNN eager (img/s) |
|---|----------------------|------------------------|---------------------|
| 1 | 526 | **682** | 135 |
| 4 | **588** | 604 | 125 |
| 8 | 547 | 574 | — |

**L2 interpretation**: Even at B=8, the working set (8 × ~1.5MB per sample ≈ 12MB) fits in Ada's 96MB L2. Batch provides only 12% throughput gain (amortized weight loads) because there's no L2 capacity pressure. On Spark (~24MB), B=4 already pushes the working set near L2 capacity — weight caching should show a larger relative gain. This is the Spark S4 experiment.

### 3.7 Tile Sensitivity (31×31/128ch, M sweep, Cb=32 fixed) — ✅

| M | Shmem | Blocks | Blocks/SM | Latency |
|---|-------|--------|-----------|---------|
| 4 | 60.1 KB | 784 | 5.5 | **0.249 ms** |
| 8 | 60.1 KB | 196 | 1.4 | 0.297 ms |
| 16 | 60.1 KB | 64 | 0.45 | 0.681 ms |

**Finding**: M=4 fastest on Ada (784 blocks saturates 142 SMs). Expected on Spark (48 SMs): M=8 should be optimal (196 blocks → 4.1/SM is adequate). This SM-count-dependent optimal tile is a supporting regime finding.

## 4. Paper Text Update Status

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | ✅ Updated | CUDA events numbers, system-level integration finding |
| §4 (PyTorch Integration) | ✅ Updated | CUDA stream correctness paragraph |
| §5.1 (Methodology) | ✅ Updated | CUDA events per-iteration, bulk timing caveat |
| §5.3 (E2E tables) | ✅ Updated | CUDA events, reduce-overhead rows, 768/32 finding |
| §5.5 (Baselines) | ✅ Updated | All latencies, 768/32 0.87ms |
| §7.3 (Implications) | ✅ Updated | RepLKNet 1.62ms |
| §7.4 (System-Level) | ✅ NEW | Dispatch overhead vs CUDA graphs |
| §8 (Conclusion) | ✅ Updated | Headline numbers |
| **§2.7 (L2 residency analysis)** | **⬜ TODO** | **New subsection — extract L2 metrics from NCU, compare Ada vs Spark** |
| **§5.2 (Table 1 per-kernel)** | **⬜ TODO** | **Expand to 6 configs × 3 methods, add L2 hit column** |
| **§5.4 (NCU analysis)** | **⬜ TODO** | **Center on L2 hit rates as explanatory variable** |
| **§5.6 (Batch scaling)** | **⬜ TODO** | **Add Spark B=1,2,4 column** |
| **§5.7 (Tile sensitivity)** | **⬜ TODO** | **Add Spark column** |
| **§7.1 (Portability)** | **⬜ TODO** | **Replace with L2 residency principle** |

## 5. Remaining Ada Work — Analysis Only (No New Experiments)

### Step 5.1: Install matplotlib + generate roofline plot (~15 min)

```bash
pip install matplotlib
python plot_roofline.py --ncu_csv results/ada_ncu_metrics.csv \
    --output results/ada_roofline_plot.png \
    --fp16_peak_tflops 91.1 --bandwidth_gbps 960
```

### Step 5.2: Extract L2 hit rates from NCU CSVs (~15 min)

The NCU CSV files in `results/ncu/` already contain L2 metrics. Extract `lts__t_sectors` or `l1tex__t_sectors` for each kernel profile. Add to a table:

| Config | Custom L2 hit (%) | cuDNN NCHW L2 hit (%) | cuDNN NHWC L2 hit (%) |
|--------|-------------------|----------------------|----------------------|
| 9×9, 1536ch | 81.7 | 51.6 | (from NCU) |
| 7×7, 768ch | 75.9 | 54.9 | (from NCU) |
| 31×31, 128ch | 95.9 | 76.5 | (from NCU) |
| 27×27, 512ch | 93.0 | 41.6 | (from NCU) |

This table becomes the quantitative backbone of the revised §7.1.

Run:
```bash
python analyze_ncu.py --ncu_dir results/ncu/ --output results/ada_ncu_l2_metrics.csv --metric l2
```

If `analyze_ncu.py` doesn't support `--metric l2`, manually extract from the CSV headers.

### Step 5.3: Build cross-platform L2 comparison table

Once Spark S6 (L2 NCU extraction) is done, merge Ada + Spark L2 data:

| Config | Ada L2 (MB) | Ada Custom L2 hit | Spark L2 (MB) | Spark Custom L2 hit | Δ hit |
|--------|------------|-------------------|---------------|--------------------|--------|
| 9×9, 1536ch | 96 | 81.7% | ~24 | (from Spark NCU) | ? |
| 31×31, 128ch | 96 | 95.9% | ~24 | (from Spark NCU) | ? |
| 27×27, 512ch | 96 | 93.0% | ~24 | (from Spark NCU) | ? |

This is the paper's central evidence table for the L2 residency principle.

## 6. Output Files

```
results/
├── ada_cudnn_layouts.csv                    ✅
├── ada_triton_results.csv                   ✅
├── ada_ncu_metrics.csv                      ✅
├── ada_ncu_tables.md                        ✅
├── ada_ncu_l2_metrics.csv                   ⬜ (Step 5.2)
├── ada_roofline_plot.png                    ⬜ (Step 5.1)
├── ada_cross_platform_l2.csv                ⬜ (Step 5.3, blocks on Spark S6)
├── ada_nsys_compile_fusion.log              ✅
├── ada_1536_20_model_half.log               ✅
├── ada_1536_20_real_model_half_b4.log       ✅
├── ada_1536_20_real_model_half_b8.log       ✅
├── ada_768_32_model_half_b1.log             ✅
├── ada_replknet_model_half_b1.log           ✅
├── ada_baselines_model_half.log             ✅
├── ada_baselines_autocast.log               ✅
├── ada_sensitivity_31x31.csv                ✅
├── ada_sensitivity_31x31.log                ✅
└── ncu/                                     ✅ (15/17 profiles)
```

## 7. Known Pitfalls

1. **Stale ninja lock**: `rm -rf ~/.cache/torch_extensions/py310_cu128/depthwise_*`
2. **torch.compile reduce-overhead on RepLKNet**: First compile 10-20 min. Cache in `/tmp/torchinductor_chenchon/`.
3. **GPU 0 occupied**: Always `CUDA_VISIBLE_DEVICES=1`.
4. **v2 kernel**: Documented dead-end behind `DEPTHWISE_9X9_VARIANT=v2` env toggle. Do not promote. See §9 of previous plan version for full NCU forensics.

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