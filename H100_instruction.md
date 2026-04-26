# H100 — Throughput-Oriented Experiments

## 1. What This Is

This repo contains custom CUDA kernels for **large-kernel depthwise convolution** — the channel-independent convolution used in ConvMixer and RepLKNet. We are revising a paper submitted to **ACM TACO** that received "major revision." The reviewers' main complaints:

1. **"No novelty"** — techniques (shmem tiling, double buffering) are well-known
2. **"cuDNN-only baseline"** — need TVM/Triton comparisons
3. **"Single GPU (RTX 6000 Ada)"** — need cross-platform evidence
4. **"Batch=1 only"** — need batch scaling study
5. **"Insufficient performance analysis"** — need roofline, occupancy, instruction mix

The revised paper thesis is: **large-kernel depthwise convolution requires regime-aware implementation** — the optimal mapping changes with batch size, hardware, and kernel size. H100 is our **throughput-oriented / datacenter** platform.

## 2. The Key Technical Insight: GEMV → GEMM Transition

This is the most important section. Understanding this drives the entire H100 experiment.

### 2.1 Current Kernel Design

Each block handles **64 channels** (32 channels packed as half2 per warp). For a single spatial output position, the computation is:

```
Weight: [64, K²]    (64 channel filters, K² taps each, stored in shared memory)
Input:  [K², 1]     (K² input values for 1 spatial position of 1 batch item)
Output: [64, 1]     (64 output channel values)
```

This is a **GEMV** (matrix-vector multiply). Scalar FMA is the natural tool.

### 2.2 With Batching

With batch size B, the SAME spatial position across B images gives:

```
Weight: [64, K²]    (same weights — reused across batch)
Input:  [K², B]     (K² input values × B batch items)
Output: [64, B]     (64 channels × B items)
```

This is a **GEMM**. For 9×9 kernel with B=16: `[64, 81] × [81, 16] → [64, 16]`.

This maps perfectly onto Tensor Core MMA tiles:
- Hopper `wgmma.mma_async` supports m64×n16×k16 (FP16)
- Our [64, 81] weight × [81, 16] input can be decomposed into ceil(81/16)=6 MMA operations
- Each MMA: [64, 16] × [16, 16] → accumulates into [64, 16]

### 2.3 Why This Matters for the Paper

This transition from GEMV (batch=1) to GEMM (large batch) means:
- **batch=1**: Scalar FMA with shared-memory tiling is optimal (our current kernels)
- **large batch**: Tensor Core MMA should be superior because it replaces many scalar FMA instructions with hardware-accelerated matrix multiply, reducing instruction count per useful MAC

Finding the **crossover batch size** where Tensor Cores beat scalar FMA is the central experimental result for H100.

## 3. Repo Structure

```
.
├── convmixer/                    # ConvMixer experiment
│   ├── benchmark.py              # Main benchmark script (currently batch=1 only)
│   ├── convmixer.py              # Model definition + checkpoint loading
│   └── kernels/
│       ├── depthwise_integration.cu    # CUDA kernels (3 variants)
│       ├── depthwise_integration.hpp   # Header
│       └── depthwise_torch_ops.py      # torch.library registration + Python module
├── replknet/                     # RepLKNet experiment
│   ├── benchmark.py              # Main benchmark script
│   └── kernels/
│       ├── depthwise_replknet_nhwc_integration.cu   # Integration wrapper
│       ├── depthwise_replknet_nhwc_integration.hpp
│       ├── depthwise_replknet_nhwc_ops.py           # torch.library registration
│       ├── depthWise_ch_128_k_31_t_56_half_forward.cu  # 31×31, 128ch, 56×56
│       ├── depthWise_ch_256_k_29_t_28_half_forward.cu  # 29×29, 256ch, 28×28
│       ├── depthWise_ch_512_k_27_t_14_half_forward.cu  # 27×27, 512ch, 14×14
│       ├── depthWise128_56_k31.hpp
│       ├── depthWise256_28_k29.hpp
│       └── depthWise512_14_k27.hpp
└── H100_instruction.md           # This file
```

## 4. Detailed Kernel Design Reference

### 4.1 ConvMixer 9×9 Kernel (the primary experimental target)

File: `convmixer/kernels/depthwise_integration.cu`, function `depthwise_32_1536_9_kernel`

- **Grid**: (4, 4, 24) = (32/8, 32/8, 1536/64) — 4×4 spatial tiles, 24 channel blocks
- **Threads**: 512 = 16 warps × 32 lanes
- **Launch bounds**: `__launch_bounds__(512, 2)` — 2 blocks per SM
- **Channel grouping**: 64 channels per block, each warp lane handles 2 channels (1 half2)
- **Shared memory layout**:
  - Weight cache: 9×9×64 half = 5184 half = 10,368 bytes
  - Input double buffer: 2 × 4×16×32 half2 = 2 × 4096 half = 16,384 bytes
  - Total: ~26,752 bytes per block
- **Spatial tiling**: Each warp computes 4 output rows × 1 output column
- **cp.async**: Double-buffered input loading via `cp.async.ca.shared.global`
- **Weight reuse**: Weight stays in shmem for all spatial positions in the tile

Key observation: The inner loop does 9×9=81 FMA operations per output pixel per channel pair (half2). With 512 threads computing 4×8=32 output positions per block, that's 81×32 = 2592 FMAs per block per channel-pair. All 16 warps share the same weight tile in shmem.

### 4.2 ConvMixer 7×7 Kernel

File: same file, function `depthwise_32_768_7_kernel`
- 1024 threads, 128 channels per block
- No double buffering (smaller kernel, simpler)

### 4.3 RepLKNet Kernels

Separate .cu files for each stage:
- 31×31: Uses "colmerge 8×12" approach — processes 8 output rows, merges 12 columns
- 29×29: Same colmerge approach
- 27×27: "naive64 half2" — simpler approach, 64 channels per block

### 4.4 Known Bug: Host-Loop Batching

**`convmixer/kernels/depthwise_torch_ops.py` lines 373-387**: The Python forward() method loops `for b in range(B)` and launches one kernel per batch item. This MUST be fixed before any batch/throughput claims. See Section 6.

## 5. Experiment Plan for H100

### 5.1 Hardware Discovery (Do This First)

```python
import torch
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
print(f"SM count: {torch.cuda.get_device_properties(0).multi_processor_count}")
print(f"Global memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CuDNN: {torch.backends.cudnn.version()}")
```

H100 should report: SM_90, 132 SMs (SXM) or 114 SMs (PCIe), ~80GB HBM3.

Also check:
```bash
nvidia-smi
nvcc --version
```

### 5.2 Compile and Verify Existing Kernels

```bash
cd convmixer
python benchmark.py --model 1536_20 --skip_accuracy --device cuda:0 --num_runs 20
python benchmark.py --model 768_32 --skip_accuracy --device cuda:0 --num_runs 20

cd ../replknet
python benchmark.py --device cuda:0 --num_runs 20
```

### 5.3 Baseline Benchmarks (cuDNN Best-Layout)

Same as SPARK branch — test cuDNN with all layouts, report the best:

```python
# Test both NCHW and NHWC for each config
# Report whichever gives lower latency
# This is the fair "cuDNN (best)" baseline
```

### 5.4 Triton Baseline

Write a Triton depthwise convolution kernel (see SPARK_instruction.md Section 5.5 for template). Let Triton auto-tune on H100.

### 5.5 Fix Batched Kernel Launch (CRITICAL)

**This must be done before ANY batch scaling or throughput claims.**

The current code launches one kernel per batch item via a Python loop. Fix this:

**Preferred approach — grid-based batching:**

Modify `depthwise_32_1536_9_kernel` signature:
```cuda
__global__ void __launch_bounds__(512, 2)
depthwise_32_1536_9_kernel_batched(
    const half* __restrict__ inputTensor,   // [B, H, W, C]
    const half* __restrict__ weight,        // [K, K, C]
    half* __restrict__ outputTensor,         // [B, H, W, C]
    int B,                                    // batch size
    int HWC_stride                            // = H * W * C (stride between batch items)
)
{
    // Existing grid: (gridH, gridW, gridCh)
    // New grid: (gridH, gridW, gridCh * B)
    int gridCh = 1536 / 64;  // = 24
    int batchIdx = blockIdx.z / gridCh;
    int chBlockIdx = blockIdx.z % gridCh;
    int blockCh = chBlockIdx * 64;
    
    // Offset input/output by batch
    const half* batchInput = inputTensor + batchIdx * HWC_stride;
    half* batchOutput = outputTensor + batchIdx * HWC_stride;
    
    // ... rest of kernel unchanged, just use batchInput/batchOutput instead
}
```

Grid launch:
```cuda
dim3 grid(4, 4, 24 * B);  // spatial tiles × (channel blocks × batch)
```

This gives the GPU one large grid to schedule — much better than B separate launches.

**Also fix the Python side** (`depthwise_torch_ops.py`):
- Change `forward()` to pass the full [B, H, W, C] tensor
- Remove the Python for-loop
- Update torch.library definitions to accept batched input

### 5.6 Batch Scaling Study (Scalar FMA Kernel)

After fixing batching, sweep batch sizes:

```python
batch_sizes = [1, 2, 4, 8, 16, 32]
configs = [
    ("9x9_1536_32",  1536, 9,  32),
    ("7x7_768_32",   768,  7,  32),
    ("31x31_128_56", 128,  31, 56),
    ("29x29_256_28", 256,  29, 28),
    ("27x27_512_14", 512,  27, 14),
]
```

For each (config, batch_size), measure:
- Custom scalar FMA kernel latency (CUDA events, cold L2)
- cuDNN (best layout) latency
- Triton latency
- Throughput = batch_size / latency

This establishes the scalar FMA baseline across batch sizes.

### 5.7 NCU Profiling: Instruction Analysis

Profile the scalar FMA kernel at batch=1 and batch=16:

```bash
ncu --set full -o profile_9x9_b1 \
    python -c "... run 9x9 kernel with batch=1 ..."

ncu --set full -o profile_9x9_b16 \
    python -c "... run 9x9 kernel with batch=16 ..."
```

Extract:
- `sm__inst_executed.sum` — total instructions
- `sm__inst_executed_pipe_fma.sum` — useful math (FMA)
- `sm__inst_executed_pipe_lsu.sum` — load/store
- `sm__inst_executed_pipe_adu.sum` — address generation
- `sm__sass_branch_targets.sum` — branches

Compute **math instruction ratio** = FMA / total. If this ratio is low (e.g., <30%), it proves the kernel is "instruction-limited" — there's significant overhead from address computation, branches, and load/store that a Tensor Core approach could eliminate.

### 5.8 Tensor Core Reformulation (The Core H100 Experiment)

This is the novel contribution. We reformulate depthwise conv as a GEMM for Tensor Cores.

#### 5.8.1 The Math

For batch B, at a single spatial output position (i, j), across 64 channels in one block:

```
Output[64, B] = Weight[64, K²] × Input[K², B]
```

Where:
- Weight[c, k] = filter weight for channel c at filter tap k (k = ki*K + kj)
- Input[k, b] = input value at position (i + ki - K//2, j + kj - K//2) for channel c in batch b

**Wait — there's a subtlety.** In standard depthwise conv, channel c's weight only multiplies channel c's input. So this is NOT a dense GEMM where all rows of Weight interact with all rows of Input. It's actually **64 independent dot products** (one per channel), each of length K², batched across B.

But that's exactly what MMA computes if we lay out the data correctly:
- Weight matrix [64, K²]: row c contains K² filter taps for channel c
- Input matrix [K², B]: column b contains K² input values for batch item b **for the same channel c at the same position**

The key: **within each channel, the dot product is the same across all batches** (same filter, different input). And the MMA instruction computes `C[m,n] += A[m,k] * B[k,n]` which is exactly this — it just happens that each row m (channel) only has meaningful interaction with its own input row.

Actually, reconsider: in NHWC layout, Input[K², B] means for filter tap k and batch b, we read `input[b, i+di, j+dj, blockCh + lane_ch]`. The 64 channels are packed contiguously in memory. So:
- A[m, k] = weight[channel_m, tap_k] — the filter weight
- B[k, n] = input[batch_n, pos_k, channel_m] — but channel_m varies with m!

This means **A and B are NOT independent matrices** — B's values depend on which row of A we're multiplying. This is a **batched** GEMM where each "channel" has its own B matrix.

#### 5.8.2 Practical Reformulation

The correct formulation for Tensor Cores is **batched GEMV → batched GEMM**:

For each group of 16 channels (MMA m-dimension tile):
```
A[16, K²] = weight tile for 16 channels
B_c[K², B] = input tile for these 16 channels (channel-dependent!)
C[16, B] = output tile
```

But since each channel in B has the same spatial positions (just different channel offsets), and in NHWC the channels are contiguous, we can load B as:
```
For filter tap k (position di, dj):
  B[k, b] = input[b, row+di, col+dj, ch_offset]  // 16 channels via half2×8
```

This means for MMA tile [16, 16, 16]:
- M = 16 channels
- N = min(B, 16) batch items
- K = 16 (chunked from K²=81, so 6 iterations)

Each MMA iteration loads:
- A fragment: 16×16 from Weight[16_channels, 16_taps]
- B fragment: 16×16 from Input[16_taps, 16_batch_items] for these 16 channels

**The B fragment is the tricky part**: for each of the 16 filter taps in this chunk, we need to load 16 batch items' worth of input values for 16 channels. In NHWC layout, one spatial position has all channels contiguous, so loading 16 channels at one position for one batch item is one 32-byte load. Across 16 batch items, that's 16 loads per tap, or 256 loads per k-chunk of 16 taps.

#### 5.8.3 Implementation Approach with wgmma.mma_async

Hopper's `wgmma.mma_async` operates on shared memory operands and is asynchronous:

```
Stage shared memory layout:
  Weight shmem: [64, K²] half — loaded once, reused across all spatial positions
  Input shmem:  [K², B] half per 64-channel group — double-buffered across spatial tiles
  
Pipeline:
  1. Load weight into shmem (once per block)
  2. For each spatial output tile:
     a. Prefetch input tile into shmem (cp.async)
     b. For k_chunk in range(0, K², 16):
        - wgmma.mma_async: C[64, B] += Weight[64, 16] × Input[16, B]
     c. Store output tile
```

Start with the simpler approach:
1. **First prototype**: Use `mma.sync` (Ampere-style, also works on Hopper) to verify correctness
2. **Then optimize**: Move to `wgmma.mma_async` for better throughput

**For `mma.sync` (simpler, SM_80+):**
```cuda
// m16n8k16 FP16 MMA
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
    "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
    : "=r"(d[0]), "=r"(d[1])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
      "r"(b[0]), "r"(b[1]),
      "r"(c[0]), "r"(c[1])
);
```

This does [16, 16] × [16, 8] → [16, 8] per instruction. For our [64, 81] × [81, B]:
- M dimension: 64 / 16 = 4 tiles
- K dimension: 81 / 16 = 6 tiles (pad last to 96)
- N dimension: B / 8 tiles
- Total MMAs per spatial position: 4 × 6 × ceil(B/8)

For B=16: 4 × 6 × 2 = 48 MMA instructions per spatial position for 64 channels.
Compare to scalar: 64 × 81 = 5184 FMA instructions.
**Instruction reduction: ~100×**

Even accounting for data staging overhead, this should be a massive win at B≥8.

#### 5.8.4 Step-by-Step Implementation Plan

**Step 1: Batched scalar baseline** (from Section 5.5)
- Fix grid-based batching
- Benchmark across batch sizes
- Profile instruction mix

**Step 2: MMA correctness test**
- Write a standalone .cu file (not integrated with PyTorch yet)
- Implement [64, 81] × [81, B] using `mma.sync.m16n8k16`
- Weight: loaded into registers from shared memory
- Input: loaded into shared memory from global
- Verify numerical correctness against CPU reference
- Target: 9×9 kernel, 1536 channels, 32×32 spatial, batch=16

**Step 3: MMA performance measurement**
- Benchmark the MMA kernel vs scalar kernel across batch sizes
- Find the crossover point
- Profile with NCU: MMA utilization, instruction count, memory throughput

**Step 4: wgmma.mma_async optimization (if Step 3 shows promise)**
- Move to Hopper-specific `wgmma.mma_async`
- Implement TMA (Tensor Memory Accelerator) for efficient shmem loading
- Overlap MMA with data movement via async pipeline

**Step 5: PyTorch integration**
- Register the TC kernel via torch.library
- Integrate into benchmark scripts
- End-to-end model benchmark

#### 5.8.5 Shared Memory Layout for MMA Kernel

```
Shared memory budget per block (max 228KB on H100 with opt-in):

Weight section:  [K², 64] half = 81 × 64 × 2 = 10,368 bytes
Input section:   [K², B] half per 64ch group
  For B=16:      81 × 16 × 2 = 2,592 bytes (×2 for double buffer = 5,184 bytes)
Output staging:  [64, B] half
  For B=16:      64 × 16 × 2 = 2,048 bytes

Total: ~18 KB per block for B=16. Well within limits.
```

For larger B or K, scale accordingly. H100's 228KB shmem per SM gives plenty of room.

#### 5.8.6 What If Tensor Cores Don't Win?

This is possible if:
- The data staging overhead (loading input into correct MMA fragment layout) dominates
- Memory bandwidth is the bottleneck, not instruction count (MMA doesn't help with bandwidth)
- The scalar kernel's shared memory approach already achieves near-peak bandwidth

**This is still a publishable finding.** The paper says: "We characterized the GEMV→GEMM transition and found that even on Hopper, the data staging cost for Tensor Core execution offsets the instruction reduction, because depthwise convolution remains memory-bandwidth-bound regardless of batch size."

That's a useful negative result that teaches the community something.

### 5.9 End-to-End Model Benchmarks

After kernel-level experiments, run full model benchmarks on H100:

```bash
cd convmixer
python benchmark.py --model 1536_20 --skip_accuracy --device cuda:0 --num_runs 500
python benchmark.py --model 1536_20 --skip_accuracy --device cuda:0 --num_runs 500 --baselines

cd ../replknet
python benchmark.py --device cuda:0 --num_runs 100 --per_layer
```

Also run with batch sizes > 1 (after fixing batching in benchmark.py).

## 6. Detailed Code Changes Required

### 6.1 Fix Batched Kernel Launch

See Section 5.5 for details. Files to modify:
- `convmixer/kernels/depthwise_integration.cu` — add batched kernel variant
- `convmixer/kernels/depthwise_integration.hpp` — update header
- `convmixer/kernels/depthwise_torch_ops.py` — remove Python loop, pass batched tensor
- Same for replknet kernels

### 6.2 New File: `convmixer/kernels/depthwise_mma.cu`

Tensor Core kernel implementation. Start with `mma.sync.m16n8k16`:

```cuda
// Prototype structure
__global__ void depthwise_mma_9x9_1536_32(
    const half* __restrict__ input,    // [B, 32, 32, 1536] NHWC
    const half* __restrict__ weight,   // [9, 9, 1536] KKC
    half* __restrict__ output,          // [B, 32, 32, 1536] NHWC
    int B
) {
    // Block handles: one spatial tile × 64 channels × all batch items
    // Grid: (gridH, gridW, 1536/64)
    // Each block loops over batch items in tiles of MMA_N (8 or 16)
    
    extern __shared__ half shmem[];
    // ... weight loading, input staging, MMA loop
}
```

### 6.3 New File: `benchmark_batch_sweep.py`

Same as SPARK branch — batch sweep across configs and methods.

### 6.4 New File: `benchmark_mma.py`

Standalone benchmark comparing scalar vs MMA kernels across batch sizes.
Should output:
```
Batch | Scalar (μs) | MMA (μs) | Speedup | Scalar TFLOPS | MMA TFLOPS
1     |   28.1      |   ???    |  ???    |    9.07       |   ???
2     |   ...       |   ...    |  ...    |    ...        |   ...
...
32    |   ...       |   ...    |  ...    |    ...        |   ...
```

### 6.5 New File: `benchmark_cudnn_layouts.py`

Same as SPARK branch.

### 6.6 New File: `benchmark_triton.py`

Same as SPARK branch.

### 6.7 New File: `profile_instruction_mix.sh`

NCU profiling script that extracts instruction mix data.

## 7. Expected Output Files

```
results/
├── h100_hardware_info.txt           # nvidia-smi, compute capability, etc.
├── h100_cudnn_layouts.csv           # cuDNN best-layout results
├── h100_batch_sweep_scalar.csv      # scalar FMA batch scaling
├── h100_batch_sweep_mma.csv         # MMA kernel batch scaling (if implemented)
├── h100_crossover_analysis.csv      # scalar vs MMA crossover points
├── h100_ncu_scalar_b1.csv           # instruction mix at batch=1
├── h100_ncu_scalar_b16.csv          # instruction mix at batch=16
├── h100_ncu_mma_b16.csv             # MMA instruction mix (if implemented)
├── h100_triton_results.csv          # Triton baseline
├── h100_e2e_model_results.csv       # end-to-end model latency
└── h100_roofline_data.csv           # roofline analysis data
```

## 8. Priority Order

If time is limited, do these in order:

1. **Hardware discovery** (5 min)
2. **Compile and verify existing kernels** (30 min)
3. **Fix batched kernel launch** (4 hr) — CRITICAL, blocks everything else
4. **Batch sweep with scalar FMA kernel** (2 hr) — establishes baseline
5. **NCU profiling** (2 hr) — proves instruction overhead, motivates TC approach
6. **cuDNN best-layout baseline** (1 hr)
7. **Triton baseline** (4 hr)
8. **MMA correctness prototype** (`mma.sync`) (8 hr) — core novelty
9. **MMA performance benchmark** (2 hr) — the crossover experiment
10. **wgmma.mma_async optimization** (8 hr) — if mma.sync shows promise
11. **End-to-end model benchmarks** (2 hr)
12. **Roofline analysis** (1 hr)

**Items 1-7 are must-haves. Items 8-12 are the stretch goals that make the paper strong.**

If the MMA kernel cannot be completed in time, the paper can still present:
- NCU data showing instruction overhead in scalar kernel at large batch
- Analysis showing the GEMV→GEMM transition and why Tensor Cores are the natural next step
- Batch scaling showing where the scalar kernel stops scaling efficiently

This is weaker but still addresses the "batch=1 only" and "no analysis" reviewer concerns.

## 9. What Goes in the Paper from H100 Experiments

- **Table**: "H100 Batch Scaling" — latency and throughput vs batch size, all methods
- **Figure**: "Crossover plot" — scalar vs MMA kernel latency as function of batch size
- **Table**: "Instruction Mix Comparison" — scalar vs MMA instruction counts
- **Figure**: Roofline plot for H100
- **Finding**: At batch ≥ X, Tensor Core reformulation outperforms scalar FMA by Y× because it reduces instruction count by Z× while maintaining the same memory bandwidth utilization
- **OR (negative result)**: Despite 100× instruction reduction, the MMA kernel does not outperform scalar FMA because depthwise conv remains memory-bandwidth-bound at all batch sizes on H100

## 10. Important Rules for the Paper

- **NEVER mention "naive kernels"** — only discuss shared memory kernels
- **cuDNN baseline must use its BEST layout/algorithm** — not forced NHWC
- **All timing must be kernel-only** (CUDA events), separated from Python overhead
- **Report cold-L2 results** for per-kernel benchmarks
- **The batched kernel must be properly implemented** — no host loops. If a reviewer discovers we're launching one kernel per batch item, the throughput numbers are invalidated
- **If MMA doesn't win, say so honestly** — negative results are valuable if well-analyzed
- **Do not overclaim H100-specific results as general** — present as "Hopper-class datacenter platform"
- **Separate the two stories clearly**: batch=1 latency (scalar wins) vs large-batch throughput (MMA may win). These are complementary, not contradictory
