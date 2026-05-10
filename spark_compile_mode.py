import torch, time, sys, os, json
import torchvision.models as tvm

ROOT = os.path.dirname(os.path.abspath(__file__))
THIS = os.path.join(ROOT, 'convmixer')
REPLKNET_DIR = os.path.join(ROOT, 'replknet')
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.join(REPLKNET_DIR, 'RepLKNet-pytorch'))
sys.path.insert(0, os.path.join(REPLKNET_DIR, 'kernels'))
from convmixer import load_pretrained
from kernels.depthwise_torch_ops import replace_depthwise_convs

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def time_it(fn, n_warm=50, n_meas=500):
    for _ in range(n_warm): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_meas): fn()
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

x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16
                ).to(memory_format=torch.channels_last)

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
print(f"{'Model':<32} {'default (ms)':>14} {'reduce-OH (ms)':>17} {'RO speedup':>12}")
print("-" * 78)

for name, factory in cases:
    # default
    m = factory()
    md = torch.compile(m, mode='default')
    with torch.no_grad():
        for _ in range(30): md(x)
        t_d = time_it(lambda: md(x))
    del m, md; torch.cuda.empty_cache(); torch._dynamo.reset()

    # reduce-overhead
    m = factory()
    try:
        mr = torch.compile(m, mode='reduce-overhead', fullgraph=False)
        with torch.no_grad():
            for _ in range(50): mr(x)
            t_r = time_it(lambda: mr(x))
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
