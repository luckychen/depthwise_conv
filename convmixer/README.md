# ConvMixer Experiments

Custom shared-memory tiled depthwise convolution kernels for ConvMixer models.

## Models

| Model | Depthwise Kernel | Channels | Depth | Params |
|-------|------------------|----------|-------|--------|
| ConvMixer-1536/20 | 9×9 | 1536 | 20 | 52M |
| ConvMixer-768/32 | 7×7 | 768 | 32 | 21M |

## Setup

### 1. Download checkpoints

Pretrained checkpoints are from [ConvMixer releases](https://github.com/locuslab/convmixer/releases):

```bash
mkdir -p checkpoints
wget -P checkpoints/ https://github.com/locuslab/convmixer/releases/download/v1.0/convmixer_1536_20_ks9_p7.pth.tar
wget -P checkpoints/ https://github.com/locuslab/convmixer/releases/download/v1.0/convmixer_768_32_ks7_p7_relu.pth.tar
```

The code expects checkpoints at `convmixer/checkpoints/<filename>.pth.tar`. This path is set in `convmixer.py` via the `CONVMIXER_CHECKPOINTS` dict and loaded relative to the script directory.

### 2. Verify environment

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
which ninja  # must be on PATH for CUDA JIT compilation
```

## Usage

### Latency benchmark (no ImageNet data needed)

```bash
# ConvMixer-1536/20 (9×9 depthwise, 20 layers)
python benchmark.py --model 1536_20 --skip_accuracy --device cuda:0

# ConvMixer-768/32 (7×7 depthwise, 32 layers)
python benchmark.py --model 768_32 --skip_accuracy --device cuda:0

# Include baseline models (MobileNetV3-Large, EfficientNet-B0, ResNet-152)
# Baseline weights are downloaded automatically from torchvision
python benchmark.py --model 1536_20 --skip_accuracy --baselines --device cuda:0

# Fewer runs for quick smoke test
python benchmark.py --model 1536_20 --skip_accuracy --num_runs 20 --device cuda:0
```

### Accuracy validation (requires ImageNet)

ImageNet is loaded via HuggingFace `datasets`:
```python
from datasets import load_dataset
dataset = load_dataset('ILSVRC/imagenet-1k', split='validation', cache_dir=data_dir)
```

The default `--data_dir` is `./data/imagenet` (relative to `benchmark.py`). On first run, the dataset is downloaded from HuggingFace (~150 GB cache). You need:
1. A HuggingFace account with access to [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
2. `huggingface-cli login` to authenticate

```bash
# Validate ConvMixer-1536/20 accuracy on full 50K ImageNet
python benchmark.py --model 1536_20 --device cuda:0

# Custom cache directory
python benchmark.py --model 1536_20 --data_dir /path/to/imagenet/cache --device cuda:0

# With baselines (also validates MobileNetV3 etc.)
python benchmark.py --model 1536_20 --baselines --device cuda:0
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `1536_20` | ConvMixer variant: `1536_20` or `768_32` |
| `--device` | `cuda:0` | CUDA device (e.g., `cuda:0`, `cuda:1`) |
| `--num_runs` | `500` | Number of timed inference iterations |
| `--skip_accuracy` | off | Skip ImageNet accuracy validation (latency only) |
| `--baselines` | off | Include MobileNetV3, EfficientNet-B0, ResNet-152 |
| `--data_dir` | `./data/imagenet` | HuggingFace ImageNet cache directory |

## What the benchmark measures

The script creates **4 configurations** for the selected ConvMixer model:

1. **CuDNN eager** — standard PyTorch with cuDNN backend
2. **CuDNN + compile** — standard PyTorch + `torch.compile(mode='default')`
3. **Custom eager** — depthwise convolutions replaced with custom CUDA kernels
4. **Custom + compile** — custom kernels + `torch.compile(mode='default')`

Each configuration is warmed up (200 iterations) then timed over `--num_runs` iterations. All use batch size 1, FP16, and channels_last memory format.

When `--baselines` is set, three additional torch.compile'd baseline models are benchmarked:
- MobileNetV3-Large (torchvision, auto-downloaded weights)
- EfficientNet-B0 (torchvision, auto-downloaded weights)
- ResNet-152 (torchvision, auto-downloaded weights)

## Code walkthrough

- **`convmixer.py`** — Model definition. `load_pretrained()` loads a checkpoint from `checkpoints/` dir. The `CONVMIXER_CHECKPOINTS` dict maps model names to filenames.
- **`kernels/depthwise_torch_ops.py`** — Registers custom CUDA ops via `torch.library.define`/`impl`/`register_fake`. The `replace_depthwise_convs()` function walks the model and swaps `nn.Conv2d` depthwise layers with `CustomDepthwiseConv2d` modules. Weight format conversion: PyTorch `[C,1,K,K]` → custom `[K,K,C]`.
- **`kernels/depthwise_integration.cu`** — The actual CUDA kernels. JIT-compiled by `torch.utils.cpp_extension.load()` on first import.
- **`benchmark.py`** — Orchestrates everything. `build_convmixer_configs()` creates the 4 model variants. `benchmark_one()` measures latency. `validate_accuracy()` runs ImageNet evaluation.

## Expected Results (batch=1, FP16 channels_last)

### ConvMixer-1536/20

| Configuration | Latency | Speedup vs CuDNN eager |
|---------------|---------|------------------------|
| CuDNN eager | ~8.7 ms | 1.00× |
| CuDNN + compile | ~7.7 ms | 1.14× |
| Custom eager | ~3.3 ms | 2.61× |
| Custom + compile | ~1.6 ms | 5.37× |

Accuracy: 81.36% top-1 / 95.61% top-5 (full 50K ImageNet, FP16+compile)

### ConvMixer-768/32

| Configuration | Latency | Speedup vs CuDNN eager |
|---------------|---------|------------------------|
| CuDNN eager | ~2.9 ms | 1.00× |
| CuDNN + compile | ~1.6 ms | 1.82× |
| Custom + compile | ~1.4 ms | 2.00× |

Accuracy: 80.15% top-1 / 95.08% top-5 (full 50K ImageNet, FP16+compile)

Exact numbers vary by GPU; the speedup ratios should be similar.
