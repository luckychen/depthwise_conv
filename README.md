# Shared-Memory Tiled Depthwise Convolution for Large Kernels

Reproducibility code for the paper: **Shared-Memory Tiled Depthwise Convolution for Large Kernels on GPUs**.

Custom CUDA kernels for large-kernel (7×7 to 31×31) depthwise convolution that are **NHWC-native** and compatible with `torch.compile`. Achieves up to **10.4× speedup** over cuDNN for 9×9 kernels and **5.37× end-to-end** on ConvMixer-1536/20.

## Hardware Requirements

- NVIDIA GPU with SM >= 8.0 (tested on SM 8.9)
- CUDA 12.x toolkit (tested with CUDA 12.4 and 12.9)
- Sufficient GPU memory (16 GB+ recommended)

## Software Requirements

We recommend using a **conda environment** with the following packages:

```bash
conda create -n cuda python=3.10
conda activate cuda

# PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Required: ninja for JIT compilation of CUDA kernels
# The CUDA kernels are compiled at runtime via torch.utils.cpp_extension,
# which requires ninja to be installed and on your PATH.
pip install ninja

# Required for RepLKNet only
pip install timm

# Optional: for ImageNet accuracy validation
pip install datasets tqdm
```

**Important:** Make sure `ninja` is on your `PATH`. If you are using conda, activate the environment before running benchmarks so that `ninja` from the conda env is found. You can verify with:
```bash
which ninja  # should point to your conda env's bin/ninja
```

### Package versions used in our experiments

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| cuDNN | 9.10.2 |
| ninja | 1.13.0 |
| timm | 1.0.24 |
| datasets | 4.4.2 |
| numpy | 2.2.6 |
| Pillow | 12.1.0 |

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

## ImageNet Accuracy Validation (Optional)

Accuracy validation requires the ImageNet-1K validation set (50,000 images). The scripts use the **HuggingFace `datasets` library** to download and cache the data.

### How ImageNet is loaded

The ConvMixer benchmark loads ImageNet via:
```python
from datasets import load_dataset
dataset = load_dataset('ILSVRC/imagenet-1k', split='validation', cache_dir=data_dir)
```

The default cache directory is `./data/imagenet` (relative to the benchmark script). You can change it with `--data_dir`:
```bash
python benchmark.py --model 1536_20 --data_dir /path/to/imagenet/cache --device cuda:0
```

### HuggingFace access

You need a HuggingFace account with access to the [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset:
1. Create an account at https://huggingface.co
2. Accept the dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
3. Log in via CLI: `huggingface-cli login`

The first download will take some time (~150 GB cache). Subsequent runs use the local cache.

### Running accuracy validation

```bash
cd convmixer
python benchmark.py --model 1536_20 --device cuda:0 --num_runs 500

# With baselines (MobileNetV3, EfficientNet-B0, ResNet-152):
python benchmark.py --model 1536_20 --baselines --device cuda:0
```

## Detailed Experiment Instructions

See the per-experiment READMEs for full details:
- [convmixer/README.md](convmixer/README.md) — ConvMixer-1536/20 and ConvMixer-768/32
- [replknet/README.md](replknet/README.md) — RepLKNet-31B

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
```

## How CUDA Kernels Are Compiled

The custom CUDA kernels are **JIT-compiled at runtime** using `torch.utils.cpp_extension.load()`. This means:

1. **No pre-compilation step needed** — just run the Python scripts
2. **First run is slow** (~30-60s) while kernels compile; compiled binaries are cached in `~/.cache/torch_extensions/`
3. **Requires `ninja`** build system on PATH (see installation above)
4. **Requires CUDA toolkit** headers (usually installed with CUDA toolkit or PyTorch)

If JIT compilation fails, you will see a warning like:
```
Failed to load custom kernels: Ninja is required to load C++ extensions
```
This means `ninja` is not on your PATH. Make sure your conda environment is activated.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ninja is required to load C++ extensions` | `pip install ninja` and ensure conda env is activated |
| `FileNotFoundError: .../checkpoints/...` | Download checkpoints (see Quick Start above) |
| `ModuleNotFoundError: No module named 'replknet'` | `git clone https://github.com/DingXiaoH/RepLKNet-pytorch` in `replknet/` dir |
| `ModuleNotFoundError: No module named 'timm'` | `pip install timm` |
| `CUDA out of memory` | Use a GPU with >= 16 GB; try `--device cuda:0` to select a specific GPU |
| First run very slow | Normal — CUDA kernels are JIT-compiled on first run; cached afterwards |
| `ILSVRC/imagenet-1k` access denied | Log in to HuggingFace and accept dataset terms (see above) |

## Notes

- All benchmarks use **batch size 1**, **FP16**, and **channels_last** (NHWC) memory format
- Custom operators are registered via `torch.library` for `torch.compile` compatibility
- The `--skip_accuracy` / `--skip_verify` flags allow latency-only runs without ImageNet data
- Use `--device cuda:N` to select which GPU to run on (default: `cuda:0`)
- Benchmark variance is normal; use `--num_runs 500` for stable measurements
