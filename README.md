# Efficient Large-Kernel Depthwise Convolution on GPUs via Shared Memory Tiling

Reproducibility artifact for the ACM TACO paper.
Custom CUDA kernels for large-kernel (7x7 to 31x31) depthwise convolution, NHWC-native, compatible with `torch.compile`.

**Key results:** 1.10x--2.03x end-to-end speedup over cuDNN-best (including opaque-wrapped cuDNN under `reduce-overhead`) across three architectures on two GPU platforms.

## Platforms and Branches

| Branch | Platform | GPU | L2 Cache | DRAM BW |
|--------|----------|-----|----------|---------|
| `main` | Ada | RTX 6000 Ada (SM 8.9) | 96 MB | 960 GB/s |
| `SPARK` | GB10 | DGX Spark GB10 (SM 12.1) | ~24 MB | ~273 GB/s |

Both branches share the same kernel source code; benchmark scripts are identical.
Use `main` for Ada experiments, `SPARK` for GB10 experiments.

## Requirements

- NVIDIA GPU with SM >= 8.0
- CUDA 12.x toolkit
- Python 3.10+
- 16 GB+ GPU memory

```bash
# Conda environment setup
conda create -n cuda python=3.10
conda activate cuda
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ninja timm

# Optional: for ImageNet accuracy validation
pip install datasets tqdm
```

Verify `ninja` is on PATH: `which ninja`

## Quick Start (Latency Only, No Dataset Required)

### ConvMixer

```bash
cd convmixer
mkdir -p checkpoints
wget -P checkpoints/ https://github.com/locuslab/convmixer/releases/download/v1.0/convmixer_1536_20_ks9_p7.pth.tar
wget -P checkpoints/ https://github.com/locuslab/convmixer/releases/download/v1.0/convmixer_768_32_ks7_p7_relu.pth.tar

python benchmark.py --model 1536_20 --skip_accuracy --num_runs 500 --device cuda:0
python benchmark.py --model 768_32 --skip_accuracy --num_runs 500 --device cuda:0
```

### RepLKNet

```bash
cd replknet
git clone https://github.com/DingXiaoH/RepLKNet-pytorch
# Download RepLKNet-31B_ImageNet-1K_224.pth from:
# https://drive.google.com/file/d/1PYJiMszZYNrkZOeYwjccvxX8UHMALB7z

python benchmark.py --skip_verify --num_runs 500 --device cuda:0
```

First run takes ~30--60s for CUDA JIT compilation; cached afterwards in `~/.cache/torch_extensions/`.

## Reproducing Paper Tables

All scripts use CUDA events per-iteration timing, `model.half()` (FP16), batch=1, channels-last.
Use `--num_runs 500 --warmup 200` for paper-quality measurements.

| Paper Table | What | Script | Example Command |
|---|---|---|---|
| Table 1 | Pointwise layout | `benchmark_pointwise_layout.py` | `python benchmark_pointwise_layout.py --device cuda:0 --num_runs 500` |
| Table 5 | Optimization walkthrough | (built into kernel; see paper) | -- |
| Table 6 | Accuracy validation | `convmixer/benchmark.py` | `python convmixer/benchmark.py --model 1536_20 --device cuda:0` |
| Table 7 | Per-kernel cross-platform | `benchmark_cudnn_layouts.py` + `benchmark_triton.py` | `python benchmark_cudnn_layouts.py --device cuda:0 --num_runs 500` |
| Tables 8--9 | ConvMixer E2E | `benchmark_compile_modes.py` + `benchmark_cudnn_opaque.py` | `python benchmark_compile_modes.py --device cuda:0 --num_runs 500` |
| Table 10 | RepLKNet per-stage DW | `replknet/benchmark_replknet_nhwc_fp16.py` | `python replknet/benchmark_replknet_nhwc_fp16.py --device cuda:0 --num_runs 20 --per_layer` |
| Table 11 | RepLKNet E2E | `benchmark_compile_modes.py` + `benchmark_cudnn_opaque.py` | (included in compile_modes output) |
| Table 12 | In-model vs standalone | `benchmark_cudnn_layouts.py` (standalone) + `replknet/benchmark_replknet_nhwc_fp16.py` (in-model) | -- |
| Table 15 | Batch sweep | `benchmark_batch_sweep.py` + `benchmark_replknet_batch_sweep.py` | `python benchmark_batch_sweep.py --model 1536_20 --device cuda:0 --num_runs 100` |
| Table 16 | Tile sensitivity | `benchmark_sensitivity.py` | `python benchmark_sensitivity.py --device cuda:0` |

## Accuracy Validation (Optional, Requires ImageNet-1K)

```bash
# Requires HuggingFace account with ILSVRC/imagenet-1k access
huggingface-cli login
cd convmixer
python benchmark.py --model 1536_20 --device cuda:0 --num_runs 500
```

## File Structure

```
convmixer/
  benchmark.py                    # E2E benchmark + accuracy validation
  convmixer.py                    # Model definition
  kernels/
    depthwise_integration.cu      # Custom CUDA kernels (9x9, 7x7, 13x13)
    depthwise_integration.hpp
    depthwise_torch_ops.py        # torch.library operator registration

replknet/
  benchmark.py                    # E2E benchmark + accuracy validation
  benchmark_replknet_nhwc_fp16.py # Per-stage DW + per-layer profiling
  kernels/
    depthwise_replknet_nhwc_integration.cu   # Integration wrapper
    depthwise_replknet_nhwc_ops.py           # torch.library ops
    depthWise_ch_128_k_31_t_56_half_forward.cu  # 31x31 kernel
    depthWise_ch_256_k_29_t_28_half_forward.cu  # 29x29 kernel
    depthWise_ch_512_k_27_t_14_half_forward.cu  # 27x27 kernel

benchmark_compile_modes.py        # Tables 8-9, 11: E2E across compile modes
benchmark_cudnn_layouts.py        # Table 7: per-kernel NCHW vs NHWC
benchmark_cudnn_opaque.py         # Tables 8-9, 11: opaque-wrapped cuDNN
benchmark_triton.py               # Table 7: auto-tuned Triton baseline
benchmark_batch_sweep.py          # Table 15: batch sweep (ConvMixer)
benchmark_replknet_batch_sweep.py # Table 15: batch sweep (RepLKNet)
benchmark_sensitivity.py          # Table 16: tile-size sensitivity
benchmark_pointwise_layout.py     # Table 1: pointwise NCHW vs NHWC
```

## How CUDA Kernels Are Compiled

Kernels are JIT-compiled at runtime via `torch.utils.cpp_extension.load()`.
No pre-compilation step needed. Requires `ninja` on PATH.
Compiled binaries cached in `~/.cache/torch_extensions/`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ninja is required` | `pip install ninja`, ensure conda env is activated |
| Checkpoint not found | Download checkpoints (see Quick Start) |
| `No module named 'timm'` | `pip install timm` |
| `No module named 'replknet'` | `git clone https://github.com/DingXiaoH/RepLKNet-pytorch` in `replknet/` |
| CUDA out of memory | Use GPU with >= 16 GB |
| First run slow | Normal -- JIT compilation, cached afterwards |
| `imagenet-1k` access denied | HuggingFace login + accept dataset terms |

## Notes

- All benchmarks default to batch=1, FP16, channels-last (NHWC)
- Custom operators registered via `torch.library` for `torch.compile` compatibility
- `--skip_accuracy` / `--skip_verify` flags run latency-only (no ImageNet needed)
- Use `--device cuda:N` to select GPU
