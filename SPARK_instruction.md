# DGX Spark — Latency-Oriented Experiments

## 1. What This Is

This repo contains custom CUDA kernels for **large-kernel depthwise convolution** — the channel-independent convolution used in ConvMixer and RepLKNet. We are revising a paper submitted to **ACM TACO** that received "major revision." The reviewers' main complaints:

1. **"No novelty"** — techniques (shmem tiling, double buffering) are well-known
2. **"cuDNN-only baseline"** — need TVM/Triton comparisons
3. **"Single GPU (RTX 6000 Ada)"** — need cross-platform evidence, especially edge
4. **"Batch=1 only"** — need batch scaling study
5. **"Insufficient performance analysis"** — need roofline, occupancy, instruction mix

The revised paper thesis is: **large-kernel depthwise convolution requires regime-aware implementation** — the optimal mapping changes with batch size, hardware, and kernel size. DGX Spark is our **latency-oriented / compact deployment** platform.

## 2. DGX Spark Platform Context

DGX Spark is an NVIDIA GB10-based "AI supercomputer on your desk." It is:
- Compact (1.2 kg), designed for local/near-edge deployment
- Grace Blackwell architecture (ARM Grace CPU + Blackwell GPU)
- NVIDIA positions it for Isaac (robotics) and Metropolis (video analytics) — real edge workloads
- Relevant because edge/local deployment is latency-sensitive, typically batch=1 or small batch

**Do NOT call it an "embedded edge device."** Call it a "compact local deployment platform" or "near-edge AI system."

**First thing to do on this machine:** Run hardware discovery (Section 5.1) to determine exact SM version, compute capability, and available features.

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
└── SPARK_instruction.md          # This file
```

## 4. Current Kernel Design (Critical Background)

### 4.1 Core Idea: Shared Memory Tiling for NHWC Depthwise Conv

Depthwise convolution has extremely low arithmetic intensity (~0.5 FLOPs/Byte for 9×9). Each output pixel needs K² multiply-accumulates, but each input value is read from global memory. Our kernels tile input into shared memory so all warps in a block reuse the same data.

### 4.2 Channel Grouping (Important for Understanding GEMV→GEMM)

Each warp handles **32 channels** packed as `half2` (16 half2 = 32 half values per warp lane). Each block handles **64 channels** (2 warp-widths). The computation per spatial output position is:

```
Weight: [K², 64]   (64 channels × K² filter taps, in shared memory)
Input:  [K², 1]    (1 spatial position per batch item, loaded from global/shmem)
Output: [64, 1]    (64 output channels)
```

This is a **GEMV** at batch=1. But with batch=B, the same warp group computes:

```
Weight: [K², 64]   (same weights)
Input:  [K², B]    (B spatial positions from B batch items)
Output: [64, B]    (64 channels × B batch items)
```

This becomes a **GEMM** when B is large enough. This is a key insight for the paper.

### 4.3 Specific Kernel Variants in This Repo

**ConvMixer kernels** (`convmixer/kernels/depthwise_integration.cu`):
- `depthwise_32_768_7_kernel`: 32×32 spatial, 768ch, 7×7. 1024 threads, 128ch/block.
- `depthwise_32_1536_9_kernel`: 32×32 spatial, 1536ch, 9×9. 512 threads, 64ch/block, `__launch_bounds__(512, 2)`. Uses cp.async double buffering.
- `depthwise_16_1536_9_kernel`: 16×16 spatial, 1536ch, 9×9. Same design as above.

**RepLKNet kernels** (`replknet/kernels/`):
- Stage 0: 31×31 kernel, 128ch, 56×56 spatial — colmerge 8×12 approach
- Stage 1: 29×29 kernel, 256ch, 28×28 spatial — colmerge 8×12 approach
- Stage 2: 27×27 kernel, 512ch, 14×14 spatial — naive64 half2 approach
- Stage 3: 13×13, 1024ch, 7×7 — uses PyTorch native (too small for custom)

### 4.4 Known Bug: Host-Loop Batching

**`convmixer/kernels/depthwise_torch_ops.py` lines 373-387** has a Python `for b in range(B)` loop that launches one kernel per batch item. This must be fixed before any batch scaling claims. See Section 6.2.

## 5. Experiment Plan for DGX Spark

### 5.1 Hardware Discovery (Do This First)

Run this immediately to determine what we're working with:

```python
import torch
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
print(f"SM count: {torch.cuda.get_device_properties(0).multi_processor_count}")
print(f"Global memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CuDNN: {torch.backends.cudnn.version()}")
```

Also run:
```bash
nvidia-smi
nvcc --version
```

Record all of this — it goes in the paper's experimental setup section.

**Important**: If the SM version is different from SM_89 (RTX 6000 Ada), you may need to recompile kernels with the correct `-arch` flag. The JIT compilation in `torch.utils.cpp_extension.load()` should handle this automatically, but verify.

### 5.2 Compile and Verify Existing Kernels

Before any benchmarking, verify the kernels compile and produce correct results on Spark:

```bash
cd convmixer
python benchmark.py --model 1536_20 --skip_accuracy --device cuda:0 --num_runs 20
python benchmark.py --model 768_32 --skip_accuracy --device cuda:0 --num_runs 20

cd ../replknet
python benchmark.py --device cuda:0 --num_runs 20
```

If compilation fails, check the error. Common issues:
- Wrong SM arch: Add correct `-gencode` flag to `extra_cuda_cflags` in the `_load_kernels()` functions
- Missing CUDA headers: Verify CUDA toolkit path
- Shared memory limits: GB10 may have different shmem limits than SM_89

### 5.3 Baseline Benchmarks (cuDNN Best-Layout)

A critical reviewer concern: our cuDNN comparison forces NHWC, but cuDNN might be faster in NCHW for some configs. We must test ALL layouts and report cuDNN's best.

Create `benchmark_cudnn_layouts.py` that tests:
```python
layouts = ['channels_last', 'contiguous']  # NHWC vs NCHW
algorithms = [
    torch.backends.cudnn.benchmark,  # auto-select best algorithm
]
```

For each kernel config (9×9/1536ch/32×32, 7×7/768ch/32×32, 31×31/128ch/56×56, etc.):
1. Create `nn.Conv2d(C, C, K, groups=C, padding=K//2)` 
2. Test with both NCHW and NHWC (channels_last) input
3. Enable `torch.backends.cudnn.benchmark = True` to let cuDNN auto-select algorithm
4. Report the best latency across all layout/algorithm combinations
5. This becomes the "cuDNN (best)" baseline in the paper

### 5.4 TVM Baseline

Install TVM (Apache TVM) and auto-tune depthwise conv for each configuration:

```python
# Target configurations to tune:
configs = [
    (1536, 9, 32, 32),    # ConvMixer 9×9
    (768, 7, 32, 32),     # ConvMixer 7×7
    (128, 31, 56, 56),    # RepLKNet stage 0
    (256, 29, 28, 28),    # RepLKNet stage 1
    (512, 27, 14, 14),    # RepLKNet stage 2
    (1024, 13, 7, 7),     # RepLKNet stage 3
]
```

For each config:
1. Define depthwise conv in TVM relay or TE
2. Run auto-tuning (XGBTuner, at least 1000 trials per config)
3. Compile optimized module
4. Benchmark kernel-only latency

This may take hours per config. Budget accordingly.

If TVM is too complex to set up, **Triton is the minimum required baseline**:

### 5.5 Triton Baseline

Write a Triton depthwise convolution kernel. This is simpler than TVM:

```python
import triton
import triton.language as tl

@triton.jit
def depthwise_conv_kernel(
    input_ptr, weight_ptr, output_ptr,
    H, W, C, K,
    stride_h, stride_w, stride_c,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # Each program handles a tile of output
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # ... implement depthwise conv with shared memory tiling
```

Triton's auto-tuning can then search over BLOCK_H, BLOCK_W, BLOCK_C, num_warps, num_stages.

**Important**: The Triton kernel should be a fair comparison — use FP16, NHWC layout, same padding. Let Triton auto-tune the tiling parameters.

### 5.6 Batch Scaling Study

**First, fix the batching bug.** The current code in `convmixer/kernels/depthwise_torch_ops.py` uses a host loop over batch items (line 373: `for b in range(B)`). This must be changed.

Two approaches to fix batching:

**Approach A (Minimal — grid-based batch)**: Add batch dimension to the CUDA grid:
```cuda
// Change grid from (gridH, gridW, gridCh) to include batch
dim3 grid(gridH, gridW, gridCh * B);
// In kernel: int batch = blockIdx.z / gridCh; int chBlock = blockIdx.z % gridCh;
```

This requires modifying the CUDA kernels to accept batch stride and compute batch-aware offsets. The kernel body stays the same — only the indexing changes.

**Approach B (Python-side — CUDA graph)**: Use CUDA graphs to batch multiple kernel launches efficiently:
```python
# Record graph with B kernel launches
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for b in range(B):
        outputs[b] = kernel(inputs[b], weight)
# Replay graph — single CPU dispatch for all B launches
g.replay()
```

**Approach A is preferred** because it gives the GPU a single large grid to schedule, which is more efficient than B small grids. But Approach B is a quick fix if modifying CUDA code is too time-consuming.

After fixing, run batch sweep:

```python
batch_sizes = [1, 2, 4, 8, 16, 32]
```

For each batch size, for each kernel config, measure:
- Custom kernel latency
- cuDNN (best layout) latency  
- TVM/Triton baseline latency

### 5.7 NCU Profiling (Instruction Mix and Memory Analysis)

This addresses reviewer complaints about "insufficient performance analysis."

For the custom kernels and cuDNN, collect NCU profiles:

```bash
ncu --set full --target-processes all \
    -o profile_9x9_b1 \
    python -c "
import torch; ...  # minimal script that runs just the kernel
"
```

Key metrics to extract:
- `sm__inst_executed.sum` — total instructions
- `sm__inst_executed_pipe_fma.sum` — useful FMA instructions (the math)
- `sm__inst_executed_pipe_lsu.sum` — load/store instructions
- `sm__sass_branch_targets.sum` — branch instructions
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` — global memory reads
- `l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum` — global memory writes
- `sm__warps_active.avg.pct_of_peak_sustained_active` — occupancy
- `lts__t_request_hit_rate.pct` — L2 hit rate
- `sm__sass_data_bytes_mem_shared.sum` — shared memory traffic

From these, compute:
1. **Math instruction ratio** = FMA instructions / total instructions → shows instruction overhead
2. **Arithmetic intensity** = FLOPs / bytes loaded from DRAM → for roofline
3. **Achieved bandwidth** = bytes transferred / kernel time → vs peak bandwidth
4. **Achieved FLOPS** = FMA count × 2 / kernel time → vs peak FLOPS

This data feeds directly into a roofline plot in the paper.

### 5.8 Roofline Analysis

Using NCU data, create roofline plots for DGX Spark:
- X-axis: Arithmetic intensity (FLOPs/Byte)
- Y-axis: Achieved GFLOPS
- Plot each kernel config as a point
- Draw the roofline ceiling (min of compute peak, bandwidth × arithmetic intensity)

This shows visually that depthwise conv sits far on the memory-bound side, justifying our shared-memory approach.

### 5.9 tcgen05 / Tensor Core Verification

**This is exploratory — do not assume it works.** GB10 is Blackwell architecture, which may support `tcgen05` tensor core instructions.

Step 1: Check compute capability and PTX target:
```bash
nvcc --list-gpu-arch  # What architectures does this nvcc support?
```

Step 2: Write a tiny microkernel that uses `tcgen05.mma` (if supported):
```cuda
// Minimal test — does this compile and run?
asm volatile("tcgen05.mma ... ");  // exact syntax depends on PTX ISA version
```

Step 3: If it compiles and runs, measure basic throughput.

**If tcgen05 does NOT work on Spark**: That's fine. The paper says "Tensor Core study conducted on H100 only" and the Spark story focuses on direct scalar FMA kernels. This is still a valid and complete contribution.

**If tcgen05 DOES work**: This is a bonus — we can show Tensor Core behavior on both platforms.

## 6. Detailed Code Changes Required

### 6.1 New File: `benchmark_cudnn_layouts.py` (both convmixer/ and replknet/)

Standalone script that benchmarks cuDNN with all layouts. Should output a table like:

```
Config              | NCHW (ms) | NHWC (ms) | Best
9×9, 1536ch, 32×32  |    76.2   |   293.0   | NCHW
7×7, 768ch, 32×32   |    18.5   |    14.2   | NHWC
31×31, 128ch, 56×56 |   ...     |   ...     | ...
```

### 6.2 Fix Batching in `convmixer/kernels/depthwise_torch_ops.py`

The `forward()` method of `CustomDepthwiseConv2d` (line 331) currently loops `for b in range(B)`. Change to:

Option 1: Modify CUDA kernel to accept batch dimension (preferred):
- Add `int B` parameter to kernel
- Grid becomes `dim3(gridH, gridW, gridCh * B)`
- In kernel: `int batchIdx = blockIdx.z / numChBlocks; int chBlockIdx = blockIdx.z % numChBlocks;`
- Input/output offsets: add `batchIdx * H * W * C`

Option 2: At minimum, remove the Python loop and use `torch.cuda.CUDAGraph`.

### 6.3 New File: `benchmark_batch_sweep.py`

Script that runs batch sweep across all configurations:
```python
batch_sizes = [1, 2, 4, 8, 16, 32]
configs = [...]  # all kernel configs
methods = ['custom', 'cudnn_best', 'triton', 'tvm']  # available baselines
```

Output: CSV with columns `config, batch_size, method, latency_ms, throughput_imgs_per_sec`

### 6.4 New File: `benchmark_triton.py` or `benchmark_tvm.py`

Baseline implementations. At minimum one of these is required.

### 6.5 New File: `profile_ncu.sh`

Shell script that runs NCU profiling for all configurations and extracts key metrics.

## 7. Expected Output Files

After all experiments, you should have:

```
results/
├── spark_hardware_info.txt          # nvidia-smi, compute capability, etc.
├── spark_cudnn_layouts.csv          # cuDNN best-layout results
├── spark_batch_sweep.csv            # batch scaling results
├── spark_ncu_metrics.csv            # NCU profiling data
├── spark_triton_results.csv         # Triton baseline (or tvm)
├── spark_roofline_data.csv          # arithmetic intensity + achieved perf
└── spark_tcgen05_verification.txt   # tcgen05 test results (pass/fail)
```

## 8. Priority Order

If time is limited, do these in order:

1. **Hardware discovery** (5 min) — we need this for the paper
2. **Compile and verify kernels** (30 min) — confirm everything works on Spark
3. **cuDNN best-layout baseline** (1 hr) — directly addresses reviewer concern
4. **Batch sweep with current kernels** (fix batching + run) (4 hr)
5. **Triton baseline** (4 hr) — minimum additional baseline
6. **NCU profiling** (2 hr) — instruction mix and memory analysis
7. **TVM baseline** (8 hr) — nice-to-have, Triton may be sufficient
8. **Roofline analysis** (1 hr, after NCU data)
9. **tcgen05 verification** (2 hr) — exploratory

## 9. What Goes in the Paper

From the Spark experiments, the paper gets:

- **Table**: "DGX Spark Latency Comparison" — custom vs cuDNN (best) vs Triton/TVM, per kernel config
- **Table**: "Batch Scaling on DGX Spark" — how latency scales with batch size
- **Figure**: Roofline plot showing depthwise conv is memory-bound on Spark
- **Table**: "Instruction Mix Analysis" — showing math vs non-math instruction ratio
- **Paragraph**: Hardware description of DGX Spark as compact deployment platform
- **Finding**: On a compact local platform, direct shared-memory tiled kernels dominate for low-latency inference

## 10. Important Rules for the Paper

- **NEVER mention "naive kernels"** — only discuss shared memory kernels
- **NEVER call DGX Spark an "embedded edge device"** — it is a "compact local deployment platform"
- **cuDNN baseline must use its BEST layout/algorithm** — not forced NHWC
- **All timing must be kernel-only** (CUDA events or NCU), separated from Python overhead
- **Report cold-L2 results** for per-kernel benchmarks (flush cache between runs)
- **Do NOT mention specific GPU model names in claims** — present as "compact Blackwell platform" or "Grace Blackwell system"
