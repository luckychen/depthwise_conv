# DGX Spark — Experiment Plan

**Date**: 2026-05-08, updated 2026-05-09
**Status**: S1–S7 complete. **S8 (compile-mode comparison) pending — see §3 below.**
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

## 3. Follow-Up Experiment — `compile(mode='reduce-overhead')` Sweep (added 2026-05-09)

### 3.1 Why this is needed

Ada's §8 launch-overhead investigation (`RTX6000_Ada_plan.md` §8) produced two findings the
cross-platform table does not yet address:

1. **Software stack ruled out.** Re-running Ada baselines with PT 2.11.0 + cuDNN 9.19 (matched
   to Spark's stack) moved baseline latencies by ≤3%. The Spark/Ada B=1 baseline gap is not a
   software-version artifact.
2. **Launch overhead dominates baselines on Ada.** Switching ResNet-152 from `compile(mode='default')`
   to `compile(mode='reduce-overhead')` (which uses CUDA graphs internally) drops latency from
   2.66 → 1.12 ms (**2.37×**). MobileNetV3 drops 1.30 → 0.27 ms (**4.78×**), EfficientNet-B0
   drops 1.71 → 0.38 ms (**4.45×**). On Ada, **58–79% of the measured B=1 latency for
   the baselines is host-side `cudaLaunchKernel` overhead, not GPU compute.**

The current `spark_baselines_model_half.log` and `spark_convmixer_*.log` files all use
`compile(mode='default')`, which means **Spark's published baseline numbers may also have
launch-overhead headroom that we have not measured**. The Ada table currently shows Spark
"winning" the baselines (MobileNet 0.97 vs Ada 1.29, ResNet 2.23 vs Ada 2.60) — but if Spark
also has launch-overhead headroom, the gap may shrink or invert under reduce-overhead.

### 3.2 Hypothesis

GB10's NVLink-C2C interconnect already has lower per-launch host→device latency than Ada's
PCIe Gen4. So `reduce-overhead` should help **less** on Spark than on Ada:

| Model | Spark default | Predicted Spark RO | Predicted RO speedup |
|-------|--------------:|-------------------:|---------------------:|
| MobileNetV3-Large | 0.97 ms | 0.5–0.7 ms | 1.5–2× (vs Ada 4.78×) |
| EfficientNet-B0 | 1.30 ms | 0.7–0.9 ms | 1.5–2× (vs Ada 4.45×) |
| ResNet-152 | 2.23 ms | 1.5–1.8 ms | 1.2–1.5× (vs Ada 2.37×) |
| ConvMixer-1536/20 Custom+compile | 4.32 ms | ~4.3 ms | ~1× (compute-bound) |
| ConvMixer-768/32 Custom+compile | 2.09 ms | 1.5–1.8 ms | 1.2–1.4× |
| RepLKNet-31B Custom+compile | 4.67 ms | already RO | (sanity check only) |

If the predictions hold, two things follow for the paper:
- The baseline cross-platform table needs a "RO" column. Under RO, Ada beats Spark by 1.6–2×
  on every baseline (i.e., the Ada peak-FLOPs/BW advantage finally materializes once launch
  overhead is removed on both platforms).
- The Custom-vs-cuDNN ratios for ConvMixer/RepLKNet stay essentially unchanged — those models
  are already in the compute-bound regime, so they validate the paper's core claim cleanly.

### 3.3 Setup (run first, ~5 min)

```bash
source <conda init> && conda activate <spark env>
cd <repo path>
git pull origin SPARK   # pick up this updated plan and any patches
export CUDA_VISIBLE_DEVICES=0  # or whichever GB10 device id is correct on Spark

# Confirm matching stack with Ada §8 results
python -c "import torch; print(torch.__version__, 'cuDNN', torch.backends.cudnn.version())" \
    | tee results/spark_env_pt211.txt
# Expect: 2.11.0+cu128, cuDNN 91900 (or whatever the Spark host has)
```

### 3.4 Step S8 — Reduce-overhead measurement script

A small standalone script is the simplest path (avoids modifying `convmixer/benchmark.py` which
is used by both branches). Save the following as `spark_compile_mode.py` and run it.

```python
# spark_compile_mode.py
import torch, time, sys, os, json
import torchvision.models as tvm
THIS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'convmixer')
sys.path.insert(0, THIS)
from convmixer import load_pretrained
from kernels.depthwise_torch_ops import replace_depthwise_convs

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def time_it(fn, n_warm=50, n_meas=500):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_meas): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_meas * 1000

def make_convmixer(name, custom):
    m = load_pretrained(name, checkpoints_dir=os.path.join(THIS, 'checkpoints'),
                        device=device, channels_last=True)
    if custom: m = replace_depthwise_convs(m, verbose=False)
    return m.to(dtype=torch.float16, memory_format=torch.channels_last).eval()

def make_baseline(factory):
    return factory(weights=None).cuda().half().to(memory_format=torch.channels_last).eval()

x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16
                ).to(memory_format=torch.channels_last)

cases = [
    ('MobileNetV3-Large',         lambda: make_baseline(tvm.mobilenet_v3_large)),
    ('EfficientNet-B0',           lambda: make_baseline(tvm.efficientnet_b0)),
    ('ResNet-152',                lambda: make_baseline(tvm.resnet152)),
    ('ConvMixer-1536/20 CuDNN',   lambda: make_convmixer('1536_20', custom=False)),
    ('ConvMixer-1536/20 Custom',  lambda: make_convmixer('1536_20', custom=True)),
    ('ConvMixer-768/32 CuDNN',    lambda: make_convmixer('768_32',  custom=False)),
    ('ConvMixer-768/32 Custom',   lambda: make_convmixer('768_32',  custom=True)),
]

results = {}
print(f"{'Model':<32} {'default (ms)':>14} {'reduce-OH (ms)':>17} {'RO speedup':>12}")
print("-" * 78)

for name, factory in cases:
    # default
    m = factory()
    md = torch.compile(m, mode='default')
    with torch.no_grad():
        for _ in range(30): md(x)
        t_d = time_it(lambda: md(x))
    del m, md; torch.cuda.empty_cache(); torch._dynamo.reset()

    # reduce-overhead
    m = factory()
    try:
        mr = torch.compile(m, mode='reduce-overhead', fullgraph=False)
        with torch.no_grad():
            for _ in range(50): mr(x)
            t_r = time_it(lambda: mr(x))
    except Exception as e:
        t_r = float('nan')
        print(f"  reduce-overhead FAILED for {name}: {type(e).__name__}: {e}", file=sys.stderr)
    del m
    if 'mr' in dir(): del mr
    torch.cuda.empty_cache(); torch._dynamo.reset()

    sp = (t_d / t_r) if t_r == t_r else float('nan')
    print(f"{name:<32} {t_d:>14.3f} {t_r:>17.3f} {sp:>11.2f}x")
    results[name] = {'default_ms': t_d, 'reduce_overhead_ms': t_r, 'ro_speedup': sp}

with open('results/spark_compile_mode.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nWrote results/spark_compile_mode.json")
```

Run:

```bash
python spark_compile_mode.py 2>&1 | tee results/spark_compile_mode.log
```

Expected runtime: ~10–15 min total (each compile-RO warmup is the slow part, ~30 s × 7 cases).

### 3.5 Pitfalls

1. **`mode='reduce-overhead'` requires fixed input shape and no Python control flow.** ConvMixer
   may emit recompile warnings on the first few iterations — the 50-iteration warmup absorbs
   them. If you see `recompile_limit (8)` errors, increase warmup or call `torch._dynamo.reset()`
   between configs (already in the script).
2. **Custom kernel + reduce-overhead.** The custom op is registered via `torch.library` and
   should capture cleanly. If `cudagraph` capture fails for the Custom variant, fall back to
   `mode='default'` for that row and note in the log — that itself is a finding worth recording.
3. **Allocator pinning.** `mode='reduce-overhead'` allocates static input/output tensors. If
   memory grows unboundedly across iterations, set `torch._inductor.config.triton.cudagraphs = True`
   explicitly; usually unnecessary on PT 2.11.
4. **Power state at B=1.** GB10 may not boost to top clock when idle between launches. Lock
   clocks if precision matters: `sudo nvidia-smi -lgc <max>` then `-rgc` to release.
5. **First-iteration latency.** RO mode's first call captures the graph (slow). The 50-iteration
   warmup ensures we measure replay only.

### 3.6 Decision rules / what to write up

After Step S8 produces `results/spark_compile_mode.json`:

- **If predictions hold (RO speedup is small for ConvMixer/RepLKNet, large for baselines):**
  Add a "Compile mode" column to `spark_cross_platform_table.md` showing both `default` and
  `reduce-overhead` numbers. Update §4.11 of `SPARK_context.md` to note that baseline-vs-Custom
  comparisons should ideally use a regime-appropriate compile mode.

- **If RO underperforms `default` for any Custom config:** That means cudagraph capture is
  failing or stream-syncing badly with the `torch.library` op. Note it as a limitation
  (not a regression of the kernel itself).

- **If RO baselines on Spark are much slower than expected** (e.g., ResNet-152 RO > 1.8 ms):
  Suggests GB10's launch overhead is a smaller fraction of B=1 latency than predicted, i.e.,
  Spark is more compute-limited even on baselines than Ada. Worth reporting as the cleaner
  story than the launch-overhead narrative.

### 3.7 Output files for Step S8

```
results/
├── spark_env_pt211.txt                       ⬜ (Step 3.3)
├── spark_compile_mode.log           ⬜ (Step 3.4 — printed table)
└── spark_compile_mode.json          ⬜ (Step 3.4 — machine-readable)
```

---

## 4. Output Files — Final State

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