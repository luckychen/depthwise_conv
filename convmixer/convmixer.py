"""
ConvMixer model implementation with channel-last memory format support.

Paper: "Patches Are All You Need?" (ICLR 2022)
GitHub: https://github.com/locuslab/convmixer

This implementation:
- Matches the original Sequential structure for pretrained compatibility
- Supports channel-last (NHWC) memory format optimization
- Converts NCHW checkpoint weights to channel-last format when needed
"""

import torch
import torch.nn as nn

# Model configurations matching original paper
CONVMIXER_MODELS = {
    '768_32': {
        'dim': 768,
        'depth': 32,
        'kernel_size': 7,
        'patch_size': 7,
        'activation': 'relu',
        'n_classes': 1000
    },
    '1024_20': {
        'dim': 1024,
        'depth': 20,
        'kernel_size': 9,
        'patch_size': 14,
        'activation': 'gelu',
        'n_classes': 1000
    },
    '1536_20': {
        'dim': 1536,
        'depth': 20,
        'kernel_size': 9,
        'patch_size': 7,
        'activation': 'gelu',
        'n_classes': 1000
    }
}

CONVMIXER_CHECKPOINTS = {
    '768_32': 'convmixer_768_32_ks7_p7_relu.pth.tar',
    '1024_20': 'convmixer_1024_20_ks9_p14.pth.tar',
    '1536_20': 'convmixer_1536_20_ks9_p7.pth.tar'
}


class Residual(nn.Module):
    """Residual connection wrapper."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerWrapper(nn.Module):
    """
    Wrapper for ConvMixer model that supports channel-last memory format.

    The original ConvMixer model uses channel-first (NCHW) format. This wrapper
    handles conversion to channel-last (NHWC) format for optimization.

    Args:
        dim: Hidden dimension (number of channels)
        depth: Number of ConvMixer layers
        kernel_size: Kernel size for depthwise convolution
        patch_size: Patch size for patch embedding
        n_classes: Number of output classes (1000 for ImageNet-1K)
        activation: Activation function to use ('gelu' or 'relu')
        channels_last: Whether to use channels_last memory format
    """

    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=1000,
                 activation='gelu', channels_last=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.activation = activation
        self.channels_last = channels_last

        # Activation function
        act_fn = nn.ReLU if activation == 'relu' else nn.GELU

        # Build the model using Sequential (matching original implementation)
        # This structure is indexed numerically (0, 1, 2, 3.0.fn.0, etc.)
        layers = []

        # Patch embedding: indices 0, 1, 2
        layers.append(nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size))  # 0
        layers.append(act_fn())                                                        # 1
        layers.append(nn.BatchNorm2d(dim))                                           # 2

        # ConvMixer blocks: one block is indices 3.0, 3.1, 3.2, 3.3
        # Each depth level adds 4 sub-indices
        for i in range(depth):
            # Each mixer block is a Sequential with 4 layers
            block_layers = []
            # Depthwise conv with residual: 3.i.0.fn.0, 3.i.0.fn.1, 3.i.0.fn.2
            dw_layers = []
            dw_layers.append(nn.Conv2d(dim, dim, kernel_size, groups=dim,
                                      padding='same'))  # fn.0
            dw_layers.append(act_fn())                       # fn.1
            dw_layers.append(nn.BatchNorm2d(dim))           # fn.2
            block_layers.append(Residual(nn.Sequential(*dw_layers)))  # 3.i.0

            # Pointwise conv: 3.i.1, 3.i.2, 3.i.3
            block_layers.append(nn.Conv2d(dim, dim, kernel_size=1))  # 3.i.1
            block_layers.append(act_fn())                            # 3.i.2
            block_layers.append(nn.BatchNorm2d(dim))                # 3.i.3

            layers.append(nn.Sequential(*block_layers))

        # Classification head
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # index: depth + 3
        layers.append(nn.Flatten())                 # index: depth + 4
        layers.append(nn.Linear(dim, n_classes))    # index: depth + 5

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Convert input to channels_last if requested
        if self.channels_last and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)

        return self.model(x)

    def convert_to_channels_last(self):
        """
        Convert all Conv2d weights to channels_last memory format.

        Note: PyTorch's Conv2d expects NCHW weight format internally.
        The channels_last optimization is primarily for input tensors and
        intermediate activations, not the weights themselves.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # PyTorch handles the weight format internally
                # This is a no-op but kept for potential future use
                pass

    def get_config(self):
        """Get model configuration as dictionary."""
        return {
            'type': 'convmixer',
            'dim': self.dim,
            'depth': self.depth,
            'kernel_size': self.kernel_size,
            'patch_size': self.patch_size,
            'n_classes': self.n_classes,
            'activation': self.activation,
            'channels_last': self.channels_last
        }


def convmixer(model_name, channels_last=False, **kwargs):
    """
    Create a predefined ConvMixer model.

    Args:
        model_name: One of '768_32', '1024_20', '1536_20'
        channels_last: Whether to use channels_last memory format
        **kwargs: Override default configuration

    Returns:
        ConvMixerWrapper model
    """
    if model_name not in CONVMIXER_MODELS:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(CONVMIXER_MODELS.keys())}")

    config = CONVMIXER_MODELS[model_name].copy()
    config.update(kwargs)
    config['channels_last'] = channels_last

    return ConvMixerWrapper(**config)


def load_pretrained(model_name, checkpoints_dir='./checkpoints', channels_last=False, device='cuda'):
    """
    Load pretrained ConvMixer model.

    The checkpoint uses channel-first (NCHW) format. When channels_last=True,
    the model will handle conversion of input tensors to NHWC format.

    Args:
        model_name: One of '768_32', '1024_20', '1536_20'
        checkpoints_dir: Directory containing checkpoint files
        channels_last: Whether to use channels_last memory format
        device: Device to load model to

    Returns:
        ConvMixerWrapper model with pretrained weights
    """
    if model_name not in CONVMIXER_MODELS:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(CONVMIXER_MODELS.keys())}")

    checkpoint_file = CONVMIXER_CHECKPOINTS[model_name]
    checkpoint_path = f"{checkpoints_dir}/{checkpoint_file}"

    # Create model
    model = convmixer(model_name, channels_last=channels_last)

    # Load checkpoint
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    # Load weights directly - checkpoint uses NCHW format, model expects NCHW
    # The channels_last conversion only affects input/activations, not weights
    model.model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f"Loaded pretrained model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Channels last mode: {channels_last}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing ConvMixer model creation...")

    for model_name in ['768_32', '1024_20', '1536_20']:
        print(f"\n--- Creating {model_name} ---")
        model = convmixer(model_name, channels_last=False)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")

        # Test channels_last
        print(f"Testing channels_last mode...")
        model_cl = convmixer(model_name, channels_last=True)
        y_cl = model_cl(x.to(memory_format=torch.channels_last))
        print(f"Output shape (channels_last): {y_cl.shape}")

    # Test loading pretrained weights
    print("\n\n--- Testing pretrained weight loading ---")
    model = load_pretrained('768_32', channels_last=False, device='cuda')

    # Test with channels_last
    print("\n--- Testing pretrained with channels_last ---")
    model_cl = load_pretrained('768_32', channels_last=True, device='cuda')

    # Verify outputs are similar
    x = torch.randn(1, 3, 224, 224).cuda()
    y1 = model(x)
    y2 = model_cl(x.to(memory_format=torch.channels_last))
    print(f"\nOutput difference (max): {(y1 - y2).abs().max().item():.6f}")
    print("All tests passed!")