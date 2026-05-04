#!/usr/bin/env python3
"""
ConvMixer benchmark: latency, throughput, and optional ImageNet accuracy.

Supports ConvMixer-1536/20 (9x9), ConvMixer-768/32 (7x7), and baseline models.

Usage:
    # ConvMixer-1536/20 latency only
    python benchmark.py --model 1536_20 --skip_accuracy --device cuda:0

    # ConvMixer-768/32 latency only
    python benchmark.py --model 768_32 --skip_accuracy --device cuda:0

    # Include baseline models (MobileNetV3, EfficientNet-B0, ResNet-152)
    python benchmark.py --model 1536_20 --skip_accuracy --baselines --device cuda:0

    # Full accuracy validation (requires ImageNet via HuggingFace datasets)
    python benchmark.py --model 1536_20 --device cuda:0
"""

import torch
import time
import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'kernels'))

from convmixer import load_pretrained
from depthwise_torch_ops import replace_depthwise_convs


def benchmark_one(model, device, num_runs=500, warmup=200, fp16_method='autocast', batch=1):
    """Benchmark a single model, return latency in ms."""
    x = torch.randn(batch, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    if fp16_method == 'model_half':
        x = x.half()
    model.eval()

    def run_forward():
        if fp16_method == 'autocast':
            with torch.amp.autocast('cuda', dtype=torch.float16):
                return model(x)
        else:
            return model(x)

    with torch.no_grad():
        for _ in range(warmup):
            run_forward()
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(num_runs):
            run_forward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    latency = (elapsed / num_runs) * 1000
    throughput = (num_runs * batch) / elapsed  # images per second
    return latency, throughput


def get_imagenet_dataloader(data_dir, batch_size=1, num_workers=4, resize=233,
                            interpolation='bicubic'):
    """Load ImageNet validation data using HuggingFace datasets."""
    from datasets import load_dataset
    from torchvision import transforms
    from pathlib import Path

    cache_path = Path(data_dir)
    if cache_path.exists():
        print(f"Loading ImageNet from cache: {data_dir}")
        dataset = load_dataset(
            'ILSVRC/imagenet-1k',
            split='validation',
            cache_dir=data_dir,
            verification_mode='no_checks'
        )
    else:
        print("Downloading ImageNet from HuggingFace...")
        dataset = load_dataset(
            'ILSVRC/imagenet-1k',
            split='validation',
            cache_dir=data_dir
        )

    interp_mode = (transforms.InterpolationMode.BICUBIC if interpolation == 'bicubic'
                   else transforms.InterpolationMode.BILINEAR)

    class HFImageNetDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, resize_size):
            self.dataset = hf_dataset
            self.transform = transforms.Compose([
                transforms.Resize(resize_size,
                                  interpolation=interp_mode,
                                  antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return self.transform(image), item['label']

    val_dataset = HFImageNetDataset(dataset, resize)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return val_loader


@torch.no_grad()
def benchmark_throughput_real(model, device, data_dir, batch=1, fp16_method='autocast',
                               num_batches=200, warmup_batches=50):
    """Measure throughput on real ImageNet validation images, data pre-loaded to GPU."""
    loader = get_imagenet_dataloader(data_dir, batch_size=batch, num_workers=4)

    model.eval()

    # Pre-load enough batches to GPU before timing starts
    total_needed = warmup_batches + num_batches
    print(f"  Pre-loading {total_needed} batches to GPU...")
    gpu_batches = []
    for images, _ in loader:
        x = images.to(device, memory_format=torch.channels_last)
        if fp16_method == 'model_half':
            x = x.half()
        gpu_batches.append(x)
        if len(gpu_batches) >= total_needed:
            break
    torch.cuda.synchronize(device)

    def run_batch(x):
        if fp16_method == 'autocast':
            with torch.amp.autocast('cuda', dtype=torch.float16):
                return model(x)
        else:
            return model(x)

    print(f"  Warming up ({warmup_batches} batches)...")
    for x in gpu_batches[:warmup_batches]:
        run_batch(x)
    torch.cuda.synchronize(device)

    print(f"  Measuring ({num_batches} batches)...")
    measure_batches = gpu_batches[warmup_batches:warmup_batches + num_batches]
    total_images = sum(x.shape[0] for x in measure_batches)

    start = time.perf_counter()
    for x in measure_batches:
        run_batch(x)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    throughput = total_images / elapsed
    latency = (elapsed / num_batches) * 1000
    return latency, throughput


@torch.no_grad()
def validate_accuracy(model, loader, device, model_name):
    """Validate top-1/top-5 accuracy on ImageNet."""
    model.eval()
    correct_1 = correct_5 = total = 0
    print(f"\nValidating {model_name}...")

    # Warmup compile
    dummy = next(iter(loader))[0][:1].to(device, memory_format=torch.channels_last)
    with torch.amp.autocast('cuda', dtype=torch.float16):
        for _ in range(50):
            _ = model(dummy)
    torch.cuda.synchronize(device)

    chunk_loader = torch.utils.data.DataLoader(
        loader.dataset, batch_size=1000, shuffle=False,
        num_workers=8, pin_memory=True)

    start = time.time()
    for images_chunk, labels_chunk in chunk_loader:
        images_gpu = images_chunk.to(device, memory_format=torch.channels_last)
        labels_gpu = labels_chunk.to(device)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            for i in range(images_gpu.shape[0]):
                out = model(images_gpu[i:i+1])
                _, pred = out.topk(5, 1, True, True)
                if pred[0, 0] == labels_gpu[i]:
                    correct_1 += 1
                if labels_gpu[i] in pred[0]:
                    correct_5 += 1
                total += 1

        elapsed = time.time() - start
        top1 = 100.0 * correct_1 / total
        top5 = 100.0 * correct_5 / total
        print(f"  [{total:>6d}/{len(loader.dataset)}] Top-1: {top1:.2f}%  Top-5: {top5:.2f}%  "
              f"({elapsed:.0f}s, {total/elapsed:.0f} img/s)")

    top1 = 100.0 * correct_1 / total
    top5 = 100.0 * correct_5 / total
    print(f"  FINAL: Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")
    return top1, top5


def build_convmixer_configs(model_name, device, fp16_method='autocast'):
    """Build all ConvMixer configurations for benchmarking."""
    configs = []

    def maybe_half(m):
        if fp16_method == 'model_half':
            m = m.to(dtype=torch.float16, memory_format=torch.channels_last)
        return m

    # CuDNN eager
    print(f"\n  Loading {model_name} CuDNN (eager)...")
    m = load_pretrained(model_name, checkpoints_dir=os.path.join(_THIS_DIR, 'checkpoints'),
                        device=device, channels_last=True)
    m = maybe_half(m).eval()
    configs.append((f'CuDNN eager', m))

    # CuDNN + compile
    print(f"  Loading {model_name} CuDNN (compile)...")
    m = load_pretrained(model_name, checkpoints_dir=os.path.join(_THIS_DIR, 'checkpoints'),
                        device=device, channels_last=True)
    m = maybe_half(m).eval()
    m = torch.compile(m, mode='default')
    configs.append((f'CuDNN + compile', m))

    # Custom eager
    print(f"  Loading {model_name} Custom (eager)...")
    m = load_pretrained(model_name, checkpoints_dir=os.path.join(_THIS_DIR, 'checkpoints'),
                        device=device, channels_last=True)
    m = replace_depthwise_convs(m, verbose=False)
    m = maybe_half(m).eval()
    configs.append((f'Custom eager', m))

    # Custom + compile
    print(f"  Loading {model_name} Custom (compile)...")
    m = load_pretrained(model_name, checkpoints_dir=os.path.join(_THIS_DIR, 'checkpoints'),
                        device=device, channels_last=True)
    m = replace_depthwise_convs(m, verbose=False)
    m = maybe_half(m).eval()
    m = torch.compile(m, mode='default')
    configs.append((f'Custom + compile', m))

    return configs


def build_baseline_configs(device, fp16_method='autocast'):
    """Build baseline model configurations."""
    import torchvision.models as models
    configs = []

    def load_baseline(m):
        if fp16_method == 'model_half':
            m = m.to(device, dtype=torch.float16, memory_format=torch.channels_last).eval()
        else:
            # autocast: keep FP32 weights, let autocast handle per-forward
            m = m.to(device, memory_format=torch.channels_last).eval()
        return m

    print("\n  Loading MobileNetV3-Large...")
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    m = load_baseline(m)
    m = torch.compile(m, mode='default')
    configs.append(('MobileNetV3-Large + compile', m))

    print("  Loading EfficientNet-B0...")
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m = load_baseline(m)
    m = torch.compile(m, mode='default')
    configs.append(('EfficientNet-B0 + compile', m))

    print("  Loading ResNet-152...")
    m = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    m = load_baseline(m)
    m = torch.compile(m, mode='default')
    configs.append(('ResNet-152 + compile', m))

    return configs


def main():
    parser = argparse.ArgumentParser(description='ConvMixer benchmark')
    parser.add_argument('--model', default='1536_20', choices=['1536_20', '768_32'],
                        help='ConvMixer variant (default: 1536_20)')
    parser.add_argument('--data_dir', default='./data/imagenet',
                        help='Path to ImageNet data cache')
    parser.add_argument('--device', default='cuda:0', help='CUDA device')
    parser.add_argument('--num_runs', type=int, default=500,
                        help='Number of benchmark runs')
    parser.add_argument('--skip_accuracy', action='store_true',
                        help='Skip accuracy validation (latency only)')
    parser.add_argument('--baselines', action='store_true',
                        help='Include baseline models (MobileNetV3, EfficientNet, ResNet)')
    parser.add_argument('--fp16_method', default='autocast',
                        choices=['autocast', 'model_half'],
                        help='FP16 method: autocast (paper, FP32 weights) or model_half (deployment, pre-converted)')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--real_data', action='store_true',
                        help='Use real ImageNet images for throughput benchmark')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print(f"ConvMixer-{args.model} Benchmark")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}, CuDNN: {torch.backends.cudnn.version()}")
    print(f"Batch size: {args.batch}, FP16 channels_last, fp16_method={args.fp16_method}")

    # Build configs
    configs = build_convmixer_configs(args.model, device, fp16_method=args.fp16_method)

    if args.baselines:
        configs.extend(build_baseline_configs(device, fp16_method=args.fp16_method))

    # Latency benchmark
    print("\n" + "=" * 70)
    print("LATENCY BENCHMARK")
    print("=" * 70)

    results = {}
    for name, model in configs:
        print(f"\n  {name}:")
        if args.real_data:
            latency, throughput = benchmark_throughput_real(
                model, device, args.data_dir, batch=args.batch,
                fp16_method=args.fp16_method, num_batches=args.num_runs)
        else:
            latency, throughput = benchmark_one(model, device, args.num_runs,
                                                fp16_method=args.fp16_method,
                                                batch=args.batch)
        print(f"    Latency:    {latency:.2f} ms")
        print(f"    Throughput: {throughput:.1f} img/s")
        results[name] = {'latency': latency, 'throughput': throughput}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    cudnn_lat = results.get('CuDNN eager', {}).get('latency', 1.0)
    print(f"\n{'Configuration':<35} {'Latency':>10} {'Throughput':>12} {'vs CuDNN':>10}")
    print("-" * 70)
    for name in results:
        r = results[name]
        speedup = cudnn_lat / r['latency']
        print(f"{name:<35} {r['latency']:>8.2f} ms {r['throughput']:>9.1f} img/s {speedup:>9.2f}x")

    # Accuracy
    if not args.skip_accuracy:
        print("\n" + "=" * 70)
        print("ACCURACY VALIDATION (full 50K ImageNet-1K)")
        print("=" * 70)

        # ConvMixer uses resize 233 + bicubic (crop_pct=0.96)
        val_loader = get_imagenet_dataloader(args.data_dir, resize=233)
        custom_model = configs[3][1]  # Custom + compile
        validate_accuracy(custom_model, val_loader, device,
                          f'ConvMixer-{args.model} (custom+compile)')

        # Also validate baseline models if requested
        if args.baselines:
            # Baselines use standard resize 256 + bilinear
            baseline_loader = get_imagenet_dataloader(
                args.data_dir, resize=256,
                interpolation='bilinear')
            for name, model in configs[4:]:  # baseline configs start at index 4
                validate_accuracy(model, baseline_loader, device, name)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
