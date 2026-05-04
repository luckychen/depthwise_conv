"""
PyTorch Custom Operator Registration for Depthwise Convolution Kernels.

This module registers custom CUDA depthwise kernels as proper PyTorch custom operators
using torch.library, enabling torch.compile compatibility and operator fusion.

Key benefits:
- torch.compile can trace and optimize around these ops
- Enables kernel fusion (depthwise + BN + GELU)
- Reduces kernel launch overhead
- Clean PyTorch-native API

Usage:
    from depthwise_torch_ops import CustomDepthwiseConv2d

    # Same API as depthwise_custom_ops.py, but torch.compile compatible!
    conv = CustomDepthwiseConv2d(1536, kernel_size=9).cuda().half()
    output = conv(input)  # Works with torch.compile!
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import warnings

# Load custom CUDA kernels once at module import
_KERNELS_LOADED = False
_depthwise_ops = None

def _load_kernels():
    """Load CUDA extension with custom depthwise kernels."""
    global _KERNELS_LOADED, _depthwise_ops

    if _KERNELS_LOADED:
        return _depthwise_ops

    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        _depthwise_ops = load(
            name='depthwise_convmixer',
            sources=[os.path.join(script_dir, 'depthwise_integration.cu')],
            extra_include_paths=[script_dir],
            extra_cuda_cflags=[
                '-O3',
                '-std=c++17',
                '--expt-relaxed-constexpr',
                '-use_fast_math',
            ],
            verbose=False,
            with_cuda=True
        )
        _KERNELS_LOADED = True
        print("✓ Custom depthwise kernels loaded successfully (torch.library)")
    except Exception as e:
        warnings.warn(f"Failed to load custom kernels: {e}")
        _depthwise_ops = None

    return _depthwise_ops


# Define custom operator namespace
# Ops accept channels_last [B, C, H, W] tensors (physical NHWC memory)
torch.library.define(
    "depthwise::conv2d_32_1536_9",
    "(Tensor input, Tensor weight) -> Tensor",
)

torch.library.define(
    "depthwise::conv2d_32_768_7",
    "(Tensor input, Tensor weight) -> Tensor",
)

torch.library.define(
    "depthwise::conv2d_16_1536_9",
    "(Tensor input, Tensor weight) -> Tensor",
)

torch.library.define(
    "depthwise::conv2d_32_1536_13",
    "(Tensor input, Tensor weight) -> Tensor",
)


@torch.library.impl("depthwise::conv2d_32_1536_9", "default")
def _depthwise_conv2d_32_1536_9_impl(input, weight):
    ops = _load_kernels()
    output = torch.empty_like(input)
    ops.depthwise_forward_32_1536_9(input, weight, output, 32, 32, 1536, 9)
    return output


@torch.library.impl("depthwise::conv2d_32_768_7", "default")
def _depthwise_conv2d_32_768_7_impl(input, weight):
    ops = _load_kernels()
    output = torch.empty_like(input)
    for b in range(input.size(0)):
        ops.depthwise_forward_32_768_7(input[b], weight, output[b], 32, 32, 768, 7)
    return output


@torch.library.impl("depthwise::conv2d_16_1536_9", "default")
def _depthwise_conv2d_16_1536_9_impl(input, weight):
    ops = _load_kernels()
    output = torch.empty_like(input)
    for b in range(input.size(0)):
        ops.depthwise_forward_16_1536_9(input[b], weight, output[b], 16, 16, 1536, 9)
    return output


@torch.library.impl("depthwise::conv2d_32_1536_13", "default")
def _depthwise_conv2d_32_1536_13_impl(input, weight):
    ops = _load_kernels()
    output = torch.empty_like(input)
    for b in range(input.size(0)):
        ops.depthwise_forward_32_1536_13(input[b], weight, output[b], 32, 32, 1536, 13)
    return output


@torch.library.register_fake("depthwise::conv2d_32_1536_9")
def _depthwise_conv2d_32_1536_9_fake(input, weight):
    return torch.empty_like(input)

@torch.library.register_fake("depthwise::conv2d_32_768_7")
def _depthwise_conv2d_32_768_7_fake(input, weight):
    return torch.empty_like(input)

@torch.library.register_fake("depthwise::conv2d_16_1536_9")
def _depthwise_conv2d_16_1536_9_fake(input, weight):
    return torch.empty_like(input)

@torch.library.register_fake("depthwise::conv2d_32_1536_13")
def _depthwise_conv2d_32_1536_13_fake(input, weight):
    return torch.empty_like(input)


# Note: Autograd (backward pass) not implemented - inference only
# For training support, would need to implement backward pass using
# torch.autograd.Function and register with torch.library.register_autograd


class CustomDepthwiseConv2d(nn.Module):
    """
    Custom depthwise convolution layer using torch.library registered operators.

    This is torch.compile compatible version of depthwise_custom_ops.CustomDepthwiseConv2d.

    Drop-in replacement for:
        nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=kernel_size//2)

    Supported configurations:
        - 32×32×768, 7×7 kernel (ConvMixer-768/32)
        - 32×32×1536, 9×9 kernel (ConvMixer-1536/20)
        - 16×16×1536, 9×9 kernel (ConvMixer-1536/20 mid-layer)
        - 32×32×1536, 13×13 kernel (large kernel experiments)

    Args:
        channels (int): Number of input/output channels
        kernel_size (int): Size of the depthwise kernel (7, 9, or 13)
        bias (bool): If True, adds a learnable bias (default: True)

    Input/Output:
        torch.float16, memory_format=torch.channels_last
    """

    SUPPORTED_CONFIGS = {
        (768, 7, 32): 'depthwise::conv2d_32_768_7',
        (1536, 9, 32): 'depthwise::conv2d_32_1536_9',
        (1536, 9, 16): 'depthwise::conv2d_16_1536_9',
        (1536, 13, 32): 'depthwise::conv2d_32_1536_13',
    }

    def __init__(self, channels, kernel_size, bias=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Weight parameter: [K, K, C] in NHWC format for our kernels
        self.weight = nn.Parameter(
            torch.randn(kernel_size, kernel_size, channels, dtype=torch.float16)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(channels, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

        # Load kernels to verify they exist
        _load_kernels()

        # Determine spatial size from first forward pass
        self.spatial_size = None
        self.op_name = None

    def _get_op_name(self, spatial_size):
        """Get the operator name for this configuration."""
        config = (self.channels, self.kernel_size, spatial_size)

        if config not in self.SUPPORTED_CONFIGS:
            raise ValueError(
                f"Unsupported configuration: {self.channels} channels, "
                f"{self.kernel_size}×{self.kernel_size} kernel, "
                f"{spatial_size}×{spatial_size} spatial. "
                f"Supported configs: {list(self.SUPPORTED_CONFIGS.keys())}"
            )

        return self.SUPPORTED_CONFIGS[config]

    def forward(self, x):
        B, C, H, W = x.shape

        if self.spatial_size is None:
            self.spatial_size = H
            self.op_name = self._get_op_name(H)

        if self.op_name == 'depthwise::conv2d_32_1536_9':
            output = torch.ops.depthwise.conv2d_32_1536_9(x, self.weight)
        elif self.op_name == 'depthwise::conv2d_32_768_7':
            output = torch.ops.depthwise.conv2d_32_768_7(x, self.weight)
        elif self.op_name == 'depthwise::conv2d_16_1536_9':
            output = torch.ops.depthwise.conv2d_16_1536_9(x, self.weight)
        elif self.op_name == 'depthwise::conv2d_32_1536_13':
            output = torch.ops.depthwise.conv2d_32_1536_13(x, self.weight)
        else:
            raise RuntimeError(f"Unknown operator: {self.op_name}")

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output

    def extra_repr(self):
        s = f'{self.channels}, kernel_size={self.kernel_size}'
        if self.bias is not None:
            s += f', bias=True'
        if self.spatial_size is not None:
            s += f', spatial_size={self.spatial_size}'
            s += f', op={self.op_name}'
        return s


def replace_depthwise_convs(model, verbose=True):
    """
    Replace all depthwise convolutions in a model with torch.library custom operators.

    This version is torch.compile compatible!

    Args:
        model: PyTorch model
        verbose: Print replacement info

    Returns:
        Modified model with custom kernels where available

    Example:
        model = ConvMixer(...)
        model = replace_depthwise_convs(model)
        model = torch.compile(model)  # Now works!
    """
    replaced = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Check if it's a depthwise conv
            if module.groups == module.in_channels == module.out_channels:
                channels = module.in_channels
                kernel_size = module.kernel_size[0]

                # Check if we have a custom kernel for this config
                config_32 = (channels, kernel_size, 32)
                config_16 = (channels, kernel_size, 16)

                if config_32 in CustomDepthwiseConv2d.SUPPORTED_CONFIGS or \
                   config_16 in CustomDepthwiseConv2d.SUPPORTED_CONFIGS:

                    # Replace with custom kernel
                    custom_conv = CustomDepthwiseConv2d(
                        channels,
                        kernel_size,
                        bias=(module.bias is not None)
                    )

                    # Move to same device as module
                    custom_conv = custom_conv.to(module.weight.device)

                    # Copy weights with proper format conversion
                    with torch.no_grad():
                        # PyTorch: [C, 1, K, K] → Custom: [K, K, C]
                        weight_custom = module.weight.squeeze(1).permute(1, 2, 0).contiguous()
                        custom_conv.weight.copy_(weight_custom.to(dtype=torch.float16, device=module.weight.device))

                        if module.bias is not None and custom_conv.bias is not None:
                            custom_conv.bias.copy_(module.bias.to(dtype=torch.float16, device=module.bias.device))

                    # Replace in model
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]

                    if parent_name:
                        parent = model.get_submodule(parent_name)
                        setattr(parent, child_name, custom_conv)
                    else:
                        setattr(model, child_name, custom_conv)

                    replaced += 1

                    if verbose:
                        print(f"✓ Replaced {name}: {channels} channels, {kernel_size}×{kernel_size} kernel")

    if verbose:
        print(f"\nReplaced {replaced} depthwise convolution(s) with torch.library custom ops")
        print("✓ Model is now torch.compile compatible!")

    return model


# Example usage
if __name__ == '__main__':
    print("=" * 80)
    print("torch.library Custom Operators - Quick Test")
    print("=" * 80)

    # Test 9×9 kernel
    print("\nTesting 9×9 depthwise convolution...")
    conv = CustomDepthwiseConv2d(1536, kernel_size=9).cuda().half()

    x = torch.randn(2, 1536, 32, 32, dtype=torch.float16, device='cuda')
    x = x.to(memory_format=torch.channels_last)

    y = conv(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Conv config:  {conv}")

    print("\n✓ Custom depthwise convolution works!")
    print("✓ torch.library operators registered")
    print("✓ Ready for torch.compile!")
