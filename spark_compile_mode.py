import torch, time, sys, os, json
import torchvision.models as tvm
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
THIS = os.path.join(ROOT, 'convmixer')

# Import HuggingFace datasets BEFORE adding replknet to sys.path —
# replknet/RepLKNet-pytorch/datasets.py shadows the HuggingFace package.
def load_real_inputs_gpu(n=5000, data_dir='./data/imagenet'):
    try:
        from datasets import load_dataset
        from torchvision import transforms
        if not Path(data_dir).exists():
            raise FileNotFoundError(data_dir)
        print(f"Loading {n} real ImageNet validation images (batch select)...")
        dataset = load_dataset('ILSVRC/imagenet-1k', split='validation',
                               cache_dir=data_dir, verification_mode='no_checks')
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Batch-select then apply transform — avoids per-item random seeks
        batch = dataset.select(range(min(n, len(dataset))))
        imgs = [val_transform(item['image'].convert('RGB')) for item in batch]
        # Move entire stack to GPU once — zero transfer cost during timing
        gpu_stack = torch.stack(imgs).to('cuda:0', dtype=torch.float16,
                                         memory_format=torch.channels_last)
        print(f"  {len(imgs)} images on GPU, {gpu_stack.nbytes/1e9:.2f} GB\n")
        return gpu_stack
    except Exception as e:
        print(f"Could not load real data ({e}), using random inputs\n")
        return torch.randn(n, 3, 224, 224, dtype=torch.float16,
                           device='cuda:0').to(memory_format=torch.channels_last)

imgs_gpu = load_real_inputs_gpu(n=5000)   # (5000, 3, 224, 224) fp16, on GPU
N_IMGS = len(imgs_gpu)

# Safe to add replknet paths now.
REPLKNET_DIR = os.path.join(ROOT, 'replknet')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(REPLKNET_DIR, 'RepLKNet-pytorch'))
sys.path.insert(0, os.path.join(REPLKNET_DIR, 'kernels'))
from convmixer import load_pretrained
from kernels.depthwise_torch_ops import replace_depthwise_convs

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def time_it(fn, n_warm=200, n_meas=5000):
    """Warm up then measure, cycling through 5000 real GPU images. No data transfer in loop."""
    with torch.no_grad():
        for i in range(n_warm):
            fn(imgs_gpu[i % N_IMGS].unsqueeze(0))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n_meas):
            fn(imgs_gpu[i % N_IMGS].unsqueeze(0))
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_meas * 1000

def make_convmixer(name, custom):
    m = load_pretrained(name, checkpoints_dir=os.path.join(THIS, 'checkpoints'),
                        device=device, channels_last=True)
    if custom: m = replace_depthwise_convs(m, verbose=False)
    return m.to(dtype=torch.float16, memory_format=torch.channels_last).eval()

def make_baseline(factory):
    return factory(weights=None).cuda().half().to(memory_format=torch.channels_last).eval()

def make_replknet(custom):
    from replknet import create_RepLKNet31B
    from depthwise_replknet_nhwc_ops import replace_replknet_depthwise_nhwc
    CKPT = os.path.join(REPLKNET_DIR, 'RepLKNet-31B_ImageNet-1K_224.pth')
    m = create_RepLKNet31B(drop_path_rate=0.3, num_classes=1000,
                           use_checkpoint=False, small_kernel_merged=False)
    sd = torch.load(CKPT, map_location='cpu')
    sd = sd.get('state_dict', sd.get('model', sd))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.structural_reparam()
    m.deep_fuse_BN()
    m.eval()
    m = m.to(device).half().to(memory_format=torch.channels_last)
    if custom:
        m = replace_replknet_depthwise_nhwc(m, verbose=False)
    return m

cases = [
    ('MobileNetV3-Large',         lambda: make_baseline(tvm.mobilenet_v3_large)),
    ('EfficientNet-B0',           lambda: make_baseline(tvm.efficientnet_b0)),
    ('ResNet-152',                lambda: make_baseline(tvm.resnet152)),
    ('ConvMixer-1536/20 CuDNN',   lambda: make_convmixer('1536_20', custom=False)),
    ('ConvMixer-1536/20 Custom',  lambda: make_convmixer('1536_20', custom=True)),
    ('ConvMixer-768/32 CuDNN',    lambda: make_convmixer('768_32',  custom=False)),
    ('ConvMixer-768/32 Custom',   lambda: make_convmixer('768_32',  custom=True)),
    ('RepLKNet-31B CuDNN',        lambda: make_replknet(custom=False)),
    ('RepLKNet-31B Custom',       lambda: make_replknet(custom=True)),
]

results = {}
print(f"Timing: 200 warmup + 5000 measurement runs, cycling through {N_IMGS} real GPU images")
print(f"{'Model':<32} {'default (ms)':>14} {'reduce-OH (ms)':>17} {'RO speedup':>12}")
print("-" * 78)

for name, factory in cases:
    # default
    m = factory()
    md = torch.compile(m, mode='default')
    t_d = time_it(lambda x: md(x))
    del m, md; torch.cuda.empty_cache(); torch._dynamo.reset()

    # reduce-overhead
    m = factory()
    try:
        mr = torch.compile(m, mode='reduce-overhead', fullgraph=False)
        t_r = time_it(lambda x: mr(x))
    except Exception as e:
        t_r = float('nan')
        print(f"  reduce-overhead FAILED for {name}: {type(e).__name__}: {e}", file=sys.stderr)
    del m
    if 'mr' in dir(): del mr
    torch.cuda.empty_cache(); torch._dynamo.reset()

    sp = (t_d / t_r) if t_r == t_r else float('nan')
    print(f"{name:<32} {t_d:>14.3f} {t_r:>17.3f} {sp:>11.2f}x")
    results[name] = {'default_ms': t_d, 'reduce_overhead_ms': t_r, 'ro_speedup': sp}

with open('results/spark_compile_mode.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nWrote results/spark_compile_mode.json")
