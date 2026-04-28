# Shared-Memory Tiled Depthwise Convolution for Large Kernels

Reproducibility code for the paper: **Shared-Memory Tiled Depthwise Convolution for Large Kernels on GPUs**.

Custom CUDA kernels for large-kernel (7×7 to 31×31) depthwise convolution that are **NHWC-native** and compatible with `torch.compile`. Achieves up to **10.4× speedup** over cuDNN for 9×9 kernels and **5.37× end-to-end** on ConvMixer-1536/20.

## Hardware Requirements

- NVIDIA GPU with SM >= 8.0
- CUDA 12.x+ toolkit
- Sufficient GPU memory (16 GB+ recommended)

### Tested Platforms

| Platform | GPU | SM | CUDA | Memory |
|----------|-----|----|------|--------|
| RTX 6000 Ada workstation | RTX 6000 Ada | SM 8.9 | 12.4 / 12.9 | 48 GB GDDR6 |
| DGX Spark (SPARK branch) | NVIDIA GB10 | SM 12.1 | 13.0 | 128.5 GB unified |

## Software Requirements

We recommend using a **conda environment** with the following packages:

```bash
conda create -n cuda python=3.10
conda activate cuda

# PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Required: ninja for JIT compilation of CUDA kernels
pip install ninja

# Required for RepLKNet only
pip install timm

# Optional: for ImageNet accuracy validation
pip install datasets tqdm
```

**Important:** Make sure `ninja` is on your `PATH`:
```bash
which ninja  # should point to your conda env's bin/ninja
```

### Package versions used in our experiments

| Package | RTX 6000 Ada | DGX Spark |
|---------|-------------|-----------|
| Python | 3.10 | 3.12.13 |
| PyTorch | 2.9.1+cu128 | 2.11.0+cu128 |
| torchvision | 0.24.1+cu128 | 0.26.0+cu128 |
| cuDNN | 9.10.2 | 9.19.0 |
| ninja | 1.13.0 | 1.13.0 |
| timm | 1.0.24 | 1.0.26 |
| datasets | 4.4.2 | — |
| numpy | 2.2.6 | 2.4.3 |
| Pillow | 12.1.0 | 12.1.1 |

## Quick Start (Latency Only, No Data Required)

The latency benchmarks only need pretrained checkpoints (model weights), **not** ImageNet data. Use `--skip_accuracy` / `--skip_verify` to run latency-only mode.

### ConvMixer

```bash
cd convmixer
mkdir -p checkpoints

# Download checkpoints
wget -P checkpoints/ https://github.com/locuslab/convmixer/releases/download/v1.0/convmixer_1536_20_ks9_p7.pth.tar
wget -P checkpoints/ https://github.com/locuslab/convmixer/releases/download/v1.0/convmixer_768_32_ks7_p7_relu.pth.tar

# Run (first run takes ~1 min for CUDA JIT compilation, subsequent runs are instant)
python benchmark.py --model 1536_20 --skip_accuracy --num_runs 20 --device cuda:0
python benchmark.py --model 768_32 --skip_accuracy --num_runs 20 --device cuda:0
```

### RepLKNet

```bash
cd replknet

# Clone model definition
git clone https://github.com/DingXiaoH/RepLKNet-pytorch

# Download checkpoint: RepLKNet-31B_ImageNet-1K_224.pth
# From Google Drive: https://drive.google.com/file/d/1PYJiMszZYNrkZOeYwjccvxX8UHMALB7z
# Place the file as: replknet/RepLKNet-31B_ImageNet-1K_224.pth

# Run
python benchmark.py --skip_verify --num_runs 20 --device cuda:0
```

## Models

### ConvMixer

| Model | Depthwise Kernel | Channels | Depth | Params |
|-------|------------------|----------|-------|--------|
| ConvMixer-1536/20 | 9×9 | 1536 | 20 | 52M |
| ConvMixer-768/32 | 7×7 | 768 | 32 | 21M |

**ConvMixer CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `1536_20` | ConvMixer variant: `1536_20` or `768_32` |
| `--device` | `cuda:0` | CUDA device |
| `--num_runs` | `500` | Number of timed inference iterations |
| `--skip_accuracy` | off | Skip ImageNet accuracy validation (latency only) |
| `--baselines` | off | Include MobileNetV3, EfficientNet-B0, ResNet-152 |
| `--data_dir` | `./data/imagenet` | HuggingFace ImageNet cache directory |

### RepLKNet-31B

Custom kernels cover stages 0-2. Stage 3 uses PyTorch native (cuDNN is efficient for 13×13 at 7×7 spatial).

| Stage | Kernel Size | Channels | Spatial | Custom Kernel | File |
|-------|-------------|----------|---------|---------------|------|
| 0 | 31×31 | 128 | 56×56 | Yes | `depthWise_ch_128_k_31_t_56_half_forward.cu` |
| 1 | 29×29 | 256 | 28×28 | Yes | `depthWise_ch_256_k_29_t_28_half_forward.cu` |
| 2 | 27×27 | 512 | 14×14 | Yes | `depthWise_ch_512_k_27_t_14_half_forward.cu` |
| 3 | 13×13 | 1024 | 7×7 | No (PyTorch native) | — |

**Setup:** Clone the RepLKNet model repo (`git clone https://github.com/DingXiaoH/RepLKNet-pytorch`) and install `timm`. The benchmark imports `create_RepLKNet31B` from `replknet/RepLKNet-pytorch/replknet.py`. Download checkpoint from Google Drive and place as `replknet/RepLKNet-31B_ImageNet-1K_224.pth`. Without checkpoint, benchmark runs with random weights (latency still valid).

**RepLKNet CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda:0` | CUDA device |
| `--num_runs` | `100` | Number of timed inference iterations |
| `--warmup` | `50` | Number of warmup iterations |
| `--per_layer` | off | Profile each depthwise layer individually |
| `--skip_verify` | off | Skip output verification against cuDNN reference |

## Benchmark Configurations

Each benchmark creates **4 configurations** (all FP16 channels_last):

1. **CuDNN eager** — standard PyTorch with cuDNN backend
2. **CuDNN + compile** — standard PyTorch + `torch.compile`
3. **Custom eager** — depthwise convolutions replaced with custom CUDA kernels
4. **Custom + compile** — custom kernels + `torch.compile`

Comparisons are **apple-to-apple**: eager vs eager, compile vs compile.

## Additional Baselines (SPARK branch)

Scripts at repo root for cross-method comparison:

- **`benchmark_cudnn_layouts.py`** — Tests cuDNN depthwise conv with both NCHW and NHWC layouts. Reports cuDNN's best achievable latency per configuration.
- **`benchmark_triton.py`** — Triton depthwise conv kernel with auto-tuning (BLOCK_C=32/64/128, num_warps=4/8). Correctness-verified against PyTorch reference.

## ImageNet Accuracy Validation (Optional)

Accuracy validation requires the ImageNet-1K validation set (50,000 images) via HuggingFace `datasets`:

```python
from datasets import load_dataset
dataset = load_dataset('ILSVRC/imagenet-1k', split='validation', cache_dir=data_dir)
```

Prerequisites:
1. HuggingFace account with access to [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
2. `huggingface-cli login` to authenticate

```bash
cd convmixer
python benchmark.py --model 1536_20 --device cuda:0 --num_runs 500

# With baselines:
python benchmark.py --model 1536_20 --baselines --device cuda:0
```

## File Structure

```
convmixer/
├── benchmark.py              # Combined benchmark (1536/20 + 768/32 + baselines)
├── convmixer.py              # Model definition
├── checkpoints/              # Place checkpoint .pth.tar files here
└── kernels/
    ├── depthwise_integration.cu   # CUDA kernels (9×9, 7×7, 13×13)
    ├── depthwise_integration.hpp  # Header with kernel declarations
    └── depthwise_torch_ops.py     # torch.library operator registration

replknet/
├── benchmark.py              # RepLKNet-31B benchmark
├── RepLKNet-pytorch/         # Clone of model repo (git clone)
├── RepLKNet-31B_ImageNet-1K_224.pth  # Checkpoint (download from Google Drive)
└── kernels/
    ├── depthwise_replknet_nhwc_integration.cu   # Integration wrapper
    ├── depthwise_replknet_nhwc_integration.hpp
    ├── depthwise_replknet_nhwc_ops.py           # torch.library ops
    ├── depthWise_ch_128_k_31_t_56_half_forward.cu  # 31×31 kernel (stage 0)
    ├── depthWise_ch_256_k_29_t_28_half_forward.cu  # 29×29 kernel (stage 1)
    ├── depthWise_ch_512_k_27_t_14_half_forward.cu  # 27×27 kernel (stage 2)
    ├── depthWise128_56_k31.hpp
    ├── depthWise256_28_k29.hpp
    └── depthWise512_14_k27.hpp

benchmark_cudnn_layouts.py    # cuDNN NCHW vs NHWC baseline
benchmark_triton.py           # Triton depthwise conv baseline
results/                      # Benchmark output files
```

## How CUDA Kernels Are Compiled

The custom CUDA kernels are **JIT-compiled at runtime** using `torch.utils.cpp_extension.load()`:

1. **No pre-compilation step needed** — just run the Python scripts
2. **First run is slow** (~30-60s) while kernels compile; cached in `~/.cache/torch_extensions/`
3. **Requires `ninja`** build system on PATH
4. **Requires CUDA toolkit** headers

## Code Walkthrough

- **`convmixer/convmixer.py`** — Model definition. `load_pretrained()` loads checkpoint from `checkpoints/` dir.
- **`convmixer/kernels/depthwise_torch_ops.py`** — Registers custom ops via `torch.library.define`/`impl`/`register_fake`. `replace_depthwise_convs()` swaps `nn.Conv2d` depthwise layers with `CustomDepthwiseConv2d`. Weight conversion: `[C,1,K,K]` → `[K,K,C]`.
- **`convmixer/kernels/depthwise_integration.cu`** — CUDA kernels. Three variants: 32×32/768ch/7×7 (1024 threads, weights-only shmem), 32×32/1536ch/9×9 (512 threads, cp.async double-buffer), 16×16/1536ch/9×9.
- **`replknet/kernels/depthwise_replknet_nhwc_ops.py`** — torch.library registration for RepLKNet. `replace_replknet_depthwise_nhwc()` replaces depthwise layers.
- **`replknet/kernels/depthWise_ch_*.cu`** — Per-stage CUDA kernels. Stages 0-1 use colmerge 8×12 approach, stage 2 uses naive64 half2.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ninja is required to load C++ extensions` | `pip install ninja` and ensure conda env is activated |
| `FileNotFoundError: .../checkpoints/...` | Download checkpoints (see Quick Start above) |
| `ModuleNotFoundError: No module named 'replknet'` | `git clone https://github.com/DingXiaoH/RepLKNet-pytorch` in `replknet/` dir |
| `ModuleNotFoundError: No module named 'timm'` | `pip install timm` |
| `CUDA out of memory` | Use a GPU with >= 16 GB; try `--device cuda:0` to select a specific GPU |
| First run very slow | Normal — CUDA kernels are JIT-compiled on first run; cached afterwards |
| `ILSVRC/imagenet-1k` access denied | Log in to HuggingFace and accept dataset terms |

## Notes

- All benchmarks use **batch size 1**, **FP16**, and **channels_last** (NHWC) memory format
- Custom operators are registered via `torch.library` for `torch.compile` compatibility
- The `--skip_accuracy` / `--skip_verify` flags allow latency-only runs without ImageNet data
- Use `--device cuda:N` to select which GPU to run on (default: `cuda:0`)
- Benchmark variance is normal; use `--num_runs 500` for stable measurements
