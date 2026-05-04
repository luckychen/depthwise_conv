#!/usr/bin/env python3
"""
ConvMixer-768/32 real ImageNet throughput validation.

Pre-loads images to GPU, runs inference one-by-one, measures total throughput.
Latency = 1000 / throughput_img_per_sec.

Reports both accuracy and throughput for each configuration.
"""

import torch
import time
import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'convmixer'))
sys.path.insert(0, os.path.join(_THIS_DIR, 'convmixer', 'kernels'))

from convmixer import load_pretrained
from depthwise_torch_ops import replace_depthwise_convs


def load_imagenet_gpu(data_dir, device, num_images=2000):
    """Load ImageNet validation images and pre-transfer to GPU as FP16 NHWC."""
    from datasets import load_dataset
    from torchvision import transforms
    from pathlib import Path

    transform = transforms.Compose([
        transforms.Resize(233, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cache_path = Path(data_dir)
    if cache_path.exists():
        ds = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=str(cache_path))
    else:
        ds = load_dataset("ILSVRC/imagenet-1k", split="validation")

    images_gpu = []
    labels_gpu = []
    count = 0
    for sample in ds:
        img = sample['image'].convert('RGB')
        tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
        tensor = tensor.to(device, dtype=torch.float16, memory_format=torch.channels_last)
        images_gpu.append(tensor)
        labels_gpu.append(sample['label'])
        count += 1
        if count >= num_images:
            break
        if count % 500 == 0:
            print(f"    Loaded {count}/{num_images} images to GPU...")

    labels_gpu = torch.tensor(labels_gpu, device=device)
    print(f"    {count} images pre-loaded to GPU as FP16 NHWC")
    return images_gpu, labels_gpu


def run_validation(model, images_gpu, labels_gpu, device, warmup=200):
    """Run inference on pre-loaded GPU images, return accuracy and throughput."""
    num_images = len(images_gpu)

    # Warmup with first image
    with torch.no_grad():
        for _ in range(warmup):
            model(images_gpu[0])
        torch.cuda.synchronize(device)

    # Timed run through all images
    correct_1 = 0
    correct_5 = 0

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in range(num_images):
            out = model(images_gpu[i])
            _, pred = out.topk(5, 1, True, True)
            if pred[0, 0] == labels_gpu[i]:
                correct_1 += 1
            if labels_gpu[i] in pred[0]:
                correct_5 += 1

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    throughput = num_images / elapsed
    latency = elapsed / num_images * 1000  # ms
    top1 = 100.0 * correct_1 / num_images
    top5 = 100.0 * correct_5 / num_images

    return top1, top5, throughput, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_images", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--data_dir", default=os.path.expanduser("~/chConv/data/imagenet"))
    args = parser.parse_args()

    device = args.device
    ckpt_dir = os.path.join(_THIS_DIR, 'convmixer', 'checkpoints')

    print("=" * 70)
    print("ConvMixer-768/32 — Real ImageNet Throughput Validation")
    print("=" * 70)
    print(f"Device:     {torch.cuda.get_device_name(device)}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"Images:     {args.num_images}")
    print(f"Warmup:     {args.warmup}")
    print()

    # Load ImageNet images to GPU
    print("Loading ImageNet images to GPU...")
    images_gpu, labels_gpu = load_imagenet_gpu(args.data_dir, device, args.num_images)
    print()

    results = []

    configs = [
        ("cuDNN NHWC eager",              False, False, None),
        ("cuDNN NHWC +compile(default)",   False, True,  'default'),
        ("cuDNN NHWC +compile(r-o)",       False, True,  'reduce-overhead'),
        ("cuDNN NCHW eager",               False, False, None),  # special handling
        ("cuDNN NCHW +compile(default)",   False, True,  'default'),  # special handling
        ("Custom NHWC eager",              True,  False, None),
        ("Custom NHWC +compile(default)",  True,  True,  'default'),
        ("Custom NHWC +compile(r-o)",      True,  True,  'reduce-overhead'),
    ]

    for name, use_custom, do_compile, compile_mode in configs:
        is_nchw = 'NCHW' in name
        print(f"\n  [{name}] Loading...", end="", flush=True)

        torch._dynamo.reset()
        m = load_pretrained('768_32', checkpoints_dir=ckpt_dir, device=device,
                            channels_last=(not is_nchw))
        if use_custom:
            m = replace_depthwise_convs(m, verbose=False)
        m = m.to(dtype=torch.float16)
        if not is_nchw:
            m = m.to(memory_format=torch.channels_last)
        m.eval()

        if do_compile:
            m = torch.compile(m, mode=compile_mode)

        # For NCHW, convert images on the fly
        if is_nchw:
            imgs = [img.contiguous() for img in images_gpu]
        else:
            imgs = images_gpu

        print(" running...", flush=True)
        top1, top5, throughput, latency = run_validation(
            m, imgs, labels_gpu, device, warmup=args.warmup)

        print(f"    Top-1: {top1:.2f}%  Top-5: {top5:.2f}%  "
              f"Throughput: {throughput:.0f} img/s  Latency: {latency:.3f} ms")

        results.append({
            "name": name,
            "top1": top1,
            "top5": top5,
            "throughput": throughput,
            "latency": latency,
        })

        del m
        torch.cuda.empty_cache()

    print(f"\n\n{'=' * 70}")
    print("SUMMARY — ConvMixer-768/32 Real ImageNet Throughput")
    print(f"{'=' * 70}")
    print(f"{'Config':<38} {'Top-1':>6} {'img/s':>8} {'Latency':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<38} {r['top1']:>5.2f}% {r['throughput']:>7.0f} {r['latency']:>8.3f} ms")

    print(f"\n{'=' * 70}")
    print("DONE")


if __name__ == "__main__":
    main()
