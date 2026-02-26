# RepLKNet-31B Experiments

Custom shared-memory tiled depthwise convolution kernels for RepLKNet-31B large-kernel stages (31×31, 29×29, 27×27).

## Kernel Coverage

| Stage | Kernel Size | Channels | Spatial | Custom Kernel | File |
|-------|-------------|----------|---------|---------------|------|
| 0 | 31×31 | 128 | 56×56 | Yes | `depthWise_ch_128_k_31_t_56_half_forward.cu` |
| 1 | 29×29 | 256 | 28×28 | Yes | `depthWise_ch_256_k_29_t_28_half_forward.cu` |
| 2 | 27×27 | 512 | 14×14 | Yes | `depthWise_ch_512_k_27_t_14_half_forward.cu` |
| 3 | 13×13 | 1024 | 7×7 | No (PyTorch native) | — |

Stage 3 uses PyTorch's native `nn.Conv2d` in channels_last mode (cuDNN is already efficient for 13×13 at 7×7 spatial).

## Setup

### 1. Clone RepLKNet model code

The benchmark imports the model definition from the official RepLKNet repo:

```bash
git clone https://github.com/DingXiaoH/RepLKNet-pytorch
```

This creates `replknet/RepLKNet-pytorch/` which contains `replknet.py`. The benchmark adds this directory to `sys.path` and imports `create_RepLKNet31B` from it.

**Note:** RepLKNet depends on `timm`. Install it if not already present:
```bash
pip install timm
```

### 2. Download checkpoint

Download the RepLKNet-31B ImageNet-1K 224×224 checkpoint:
- **Google Drive:** https://drive.google.com/file/d/1PYJiMszZYNrkZOeYwjccvxX8UHMALB7z
- Place as: `replknet/RepLKNet-31B_ImageNet-1K_224.pth`

The code looks for the checkpoint at `os.path.join(_THIS_DIR, 'RepLKNet-31B_ImageNet-1K_224.pth')` (i.e., in the same directory as `benchmark.py`). If the checkpoint is not found, the script prints a warning and runs with random weights (latency measurements are still valid).

### 3. Verify directory structure

After setup, you should have:
```
replknet/
├── benchmark.py
├── RepLKNet-31B_ImageNet-1K_224.pth   # checkpoint
├── RepLKNet-pytorch/                   # cloned repo
│   └── replknet.py                    # model definition
└── kernels/
    └── ...                            # CUDA kernels (already included)
```

### 4. Verify environment

```bash
python -c "import torch; print(torch.__version__); import timm; print('timm OK')"
which ninja  # must be on PATH for CUDA JIT compilation
```

## Usage

### Latency benchmark

```bash
# Standard benchmark (100 runs, with output verification)
python benchmark.py --device cuda:0 --num_runs 100

# Quick smoke test (skip verification, fewer runs)
python benchmark.py --device cuda:0 --skip_verify --num_runs 20

# With per-layer depthwise profiling (shows per-stage breakdown)
python benchmark.py --device cuda:0 --per_layer
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda:0` | CUDA device (e.g., `cuda:0`, `cuda:1`) |
| `--num_runs` | `100` | Number of timed inference iterations |
| `--warmup` | `50` | Number of warmup iterations |
| `--per_layer` | off | Profile each depthwise layer individually |
| `--skip_verify` | off | Skip output verification against cuDNN reference |

## What the benchmark measures

The script creates **4 configurations** (all FP16 channels_last):

1. **Custom eager** — depthwise convolutions replaced with custom CUDA kernels
2. **Custom + compile** — custom kernels + `torch.compile(mode='reduce-overhead')`
3. **CuDNN eager** — standard PyTorch with cuDNN backend
4. **CuDNN + compile** — standard PyTorch + `torch.compile(mode='reduce-overhead')`

Unless `--skip_verify` is set, the script also compares outputs between the CuDNN reference and custom kernel models (top-1/top-5 match, max logit difference).

When `--per_layer` is set, the script hooks into each depthwise layer, captures activations, and times each layer individually. This produces the per-stage breakdown table.

## Code walkthrough

- **`benchmark.py`** — Main benchmark script. `load_model()` loads RepLKNet-31B, calls `structural_reparam()` and `deep_fuse_BN()` to merge BN into conv weights, then converts to FP16 channels_last. The checkpoint path is `os.path.join(_THIS_DIR, 'RepLKNet-31B_ImageNet-1K_224.pth')`.
- **`kernels/depthwise_replknet_nhwc_ops.py`** — Registers custom ops via `torch.library`. The `replace_replknet_depthwise_nhwc()` function replaces depthwise `nn.Conv2d` with zero-copy NHWC modules. Weight conversion: `[C,1,K,K]` → `[K,K,C]`. Input/output are channels_last tensors — the physical NHWC layout matches the kernel expectation directly (no permute needed).
- **`kernels/depthwise_replknet_nhwc_integration.cu`** — C++ wrapper that exposes kernel launch functions to Python via pybind11.
- **`kernels/depthWise_ch_*.cu`** — The actual CUDA kernels for each stage. JIT-compiled together by `torch.utils.cpp_extension.load()`.

## Expected Results (batch=1, FP16 channels_last)

### End-to-end

| Configuration | Latency | Speedup vs CuDNN eager |
|---------------|---------|------------------------|
| CuDNN eager | ~5.8 ms | 1.00× |
| CuDNN + compile | ~4.4 ms | 1.33× |
| Custom eager | ~5.5 ms | 1.07× |
| Custom + compile | ~1.5 ms | 3.79× |

Accuracy: 83.30% top-1 / 96.50% top-5 (full 50K ImageNet, FP16+compile)

### Per-stage depthwise breakdown

| Stage | Kernel | Layers | CuDNN | Custom | Speedup |
|-------|--------|--------|-------|--------|---------|
| 0 | 31×31 | 2 | 1.23 ms | 0.23 ms | 5.3× |
| 1 | 29×29 | 2 | 0.53 ms | 0.15 ms | 3.6× |
| 2 | 27×27 | 18 | 1.96 ms | 0.41 ms | 4.8× |
| **Total** | | **22** | **3.87 ms** | **0.93 ms** | **4.2×** |

Exact numbers vary by GPU; speedup ratios should be similar.
