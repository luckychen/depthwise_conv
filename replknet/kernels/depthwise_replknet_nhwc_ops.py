"""
RepLKNet Depthwise Convolution - NHWC-Native Zero-Copy Integration

Uses torch.library to register custom ops for torch.compile compatibility.

Input/output: channels_last [B, C, H, W] FP16 tensors.
The physical memory layout of channels_last is [B, H, W, C] = NHWC,
which matches the kernel expectation directly. No permute, no copy.

Kernels are launched on PyTorch's current CUDA stream (not stream 0).

Custom kernels handle stages 0-2:
  - Stage 0: 31x31, 128ch, 56x56 (colmerge 8x12)
  - Stage 1: 29x29, 256ch, 28x28 (colmerge 8x12)
  - Stage 2: 27x27, 512ch, 14x14 (naive64 half2)
Stage 3 (13x13, 1024ch, 7x7) uses PyTorch native conv2d in channels_last.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

_LOADED = False
_ops = None


def _load_kernels():
    global _LOADED, _ops
    if _LOADED:
        return _ops

    d = os.path.dirname(os.path.abspath(__file__))
    _ops = load(
        name='depthwise_replknet_nhwc_v2',
        sources=[
            os.path.join(d, 'depthwise_replknet_nhwc_integration.cu'),
            os.path.join(d, 'depthWise_ch_128_k_31_t_56_half_forward.cu'),
            os.path.join(d, 'depthWise_ch_256_k_29_t_28_half_forward.cu'),
            os.path.join(d, 'depthWise_ch_512_k_27_t_14_half_forward.cu'),
        ],
        extra_include_paths=[d],
        extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', '-use_fast_math'],
        verbose=False,
        with_cuda=True,
    )
    _LOADED = True
    return _ops


# ---- torch.library custom op definitions ----

torch.library.define(
    "replknet_dw::conv_56_128_31",
    "(Tensor input, Tensor weight, Tensor bias) -> Tensor",
)

torch.library.define(
    "replknet_dw::conv_28_256_29",
    "(Tensor input, Tensor weight, Tensor bias) -> Tensor",
)

torch.library.define(
    "replknet_dw::conv_14_512_27",
    "(Tensor input, Tensor weight, Tensor bias) -> Tensor",
)


# ---- Implementations ----

@torch.library.impl("replknet_dw::conv_56_128_31", "default")
def _conv_56_128_31_impl(input, weight, bias):
    ops = _load_kernels()
    return ops.depthwise_31x31_128_nhwc(input, weight, bias)


@torch.library.impl("replknet_dw::conv_28_256_29", "default")
def _conv_28_256_29_impl(input, weight, bias):
    ops = _load_kernels()
    return ops.depthwise_29x29_256_nhwc(input, weight, bias)


@torch.library.impl("replknet_dw::conv_14_512_27", "default")
def _conv_14_512_27_impl(input, weight, bias):
    ops = _load_kernels()
    return ops.depthwise_27x27_512_nhwc(input, weight, bias)


# ---- Fake (meta) implementations for torch.compile shape inference ----

@torch.library.register_fake("replknet_dw::conv_56_128_31")
def _conv_56_128_31_fake(input, weight, bias):
    return torch.empty_like(input)


@torch.library.register_fake("replknet_dw::conv_28_256_29")
def _conv_28_256_29_fake(input, weight, bias):
    return torch.empty_like(input)


@torch.library.register_fake("replknet_dw::conv_14_512_27")
def _conv_14_512_27_fake(input, weight, bias):
    return torch.empty_like(input)


# ---- nn.Module wrappers ----

class NHWCDepthwise31x31_128(nn.Module):
    """31x31 depthwise conv, 128ch, 56x56. Zero-copy NHWC."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(31, 31, 128, dtype=torch.float16))
        self.bias = nn.Parameter(torch.empty(128, dtype=torch.float16))
        _load_kernels()

    def forward(self, x):
        return torch.ops.replknet_dw.conv_56_128_31(x, self.weight, self.bias)


class NHWCDepthwise29x29_256(nn.Module):
    """29x29 depthwise conv, 256ch, 28x28. Zero-copy NHWC."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(29, 29, 256, dtype=torch.float16))
        self.bias = nn.Parameter(torch.empty(256, dtype=torch.float16))
        _load_kernels()

    def forward(self, x):
        return torch.ops.replknet_dw.conv_28_256_29(x, self.weight, self.bias)


class NHWCDepthwise27x27_512(nn.Module):
    """27x27 depthwise conv, 512ch, 14x14. Zero-copy NHWC."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(27, 27, 512, dtype=torch.float16))
        self.bias = nn.Parameter(torch.empty(512, dtype=torch.float16))
        _load_kernels()

    def forward(self, x):
        return torch.ops.replknet_dw.conv_14_512_27(x, self.weight, self.bias)


# Map (channels, kernel_size) -> NHWC module class
_REPLACEMENT_MAP = {
    (128, 31): NHWCDepthwise31x31_128,
    (256, 29): NHWCDepthwise29x29_256,
    (512, 27): NHWCDepthwise27x27_512,
}


def replace_replknet_depthwise_nhwc(model, verbose=True):
    """
    Replace depthwise Conv2d layers in RepLKNet with zero-copy NHWC custom modules.

    The model should already have been through structural_reparam() and deep_fuse_BN()
    so that depthwise convs have bias (from fused BN).

    Args:
        model: RepLKNet model (already reparam'd and BN-fused)
        verbose: Print replacement info

    Returns:
        Modified model with custom NHWC depthwise kernels
    """
    replaced = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Conv2d):
            continue
        if module.groups != module.in_channels or module.groups != module.out_channels:
            continue

        ch = module.in_channels
        ks = module.kernel_size[0]
        key = (ch, ks)

        if key not in _REPLACEMENT_MAP:
            continue

        custom = _REPLACEMENT_MAP[key]().to(module.weight.device)

        with torch.no_grad():
            # [C,1,K,K] -> [K,K,C]
            w = module.weight.squeeze(1).permute(1, 2, 0).contiguous().half()
            custom.weight.copy_(w)

            if module.bias is not None:
                custom.bias.copy_(module.bias.half())
            else:
                custom.bias.zero_()

        # Replace in model
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], custom)
        replaced += 1

        if verbose:
            print(f"  Replaced {name}: {ch}ch, {ks}x{ks} -> {custom.__class__.__name__}")

    if verbose:
        print(f"Replaced {replaced} depthwise conv(s) with NHWC custom kernels")

    return model
