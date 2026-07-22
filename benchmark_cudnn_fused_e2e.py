"""E5b step 2-3: opaque-wrapped in-model fused-cuDNN E2E baseline.

Replaces each ConvMixer DW block (Residual(Conv2d(groups=dim) -> act_fn -> BatchNorm2d))
with a torch.library custom op whose CUDA impl calls a prebuilt cudnn-frontend fused
graph (conv + bias + activation + scale/shift, i.e. conv+act+BN folded), matching the
strongest per-kernel fused baseline found in E5/E5b-step1. Verifies numerical correctness
against the unmodified model before timing. Measures E2E eager and torch.compile
reduce-overhead (recording whether r-o capture succeeds over the opaque op).
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "convmixer"))

import cudnn

device = "cuda"
WARMUP = 500
TIMED = 500

_GRAPH_CACHE = {}
_CUDNN_HANDLE = None


def _get_handle():
    global _CUDNN_HANDLE
    if _CUDNN_HANDLE is None:
        _CUDNN_HANDLE = cudnn.create_handle()
    # Bind the cudnn handle to PyTorch's CURRENT CUDA stream before every use.
    # Without this, cudnn executes on its own default-created stream, invisible
    # to torch.compile's reduce-overhead CUDA-graph capture/replay, producing
    # silently-wrong (NaN) outputs -- the same bug class documented in
    # context.md's "CUDA Stream Fix" for the Custom kernels (2026-05-04).
    cudnn.set_stream(_CUDNN_HANDLE, torch.cuda.current_stream().cuda_stream)
    return _CUDNN_HANDLE


def _get_fused_graph(C, K, H, W, act_kind):
    key = (C, K, H, W, act_kind)
    if key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]
    pad = K // 2
    x_proto = torch.zeros(1, C, H, W, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    w_proto = torch.zeros(C, 1, K, K, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    b_proto = torch.zeros(1, C, 1, 1, device=device, dtype=torch.float16)
    sc_proto = torch.zeros(1, C, 1, 1, device=device, dtype=torch.float16)
    sh_proto = torch.zeros(1, C, 1, 1, device=device, dtype=torch.float16)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        compute_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
    )
    xt = graph.tensor_like(x_proto)
    wt = graph.tensor_like(w_proto)
    bt = graph.tensor_like(b_proto)
    st = graph.tensor_like(sc_proto)
    sht = graph.tensor_like(sh_proto)
    y = graph.conv_fprop(image=xt, weight=wt, padding=[pad, pad], stride=[1, 1], dilation=[1, 1])
    y = graph.bias(input=y, bias=bt)
    if act_kind == "gelu":
        y = graph.gelu(input=y)
    elif act_kind == "relu":
        y = graph.relu(input=y)
    else:
        raise ValueError(act_kind)
    y = graph.mul(a=y, b=st)
    y = graph.add(a=y, b=sht)
    y.set_output(True).set_data_type(cudnn.data_type.HALF)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    ws = torch.empty(graph.get_workspace_size(), device=device, dtype=torch.uint8)
    entry = dict(graph=graph, xt=xt, wt=wt, bt=bt, st=st, sht=sht, y=y, ws=ws)
    _GRAPH_CACHE[key] = entry
    return entry


torch.library.define(
    "cudnn_fused::dw_act_bn",
    "(Tensor input, Tensor weight, Tensor bias, Tensor scale, Tensor shift, int act_kind) -> Tensor",
)


@torch.library.impl("cudnn_fused::dw_act_bn", "CUDA")
def _dw_act_bn_impl(input, weight, bias, scale, shift, act_kind):
    N, C, H, W = input.shape
    K = weight.shape[2]
    act_str = "gelu" if act_kind == 0 else "relu"
    entry = _get_fused_graph(C, K, H, W, act_str)
    y_out = torch.empty_like(input)
    handle = _get_handle()
    entry["graph"].execute(
        {entry["xt"]: input, entry["wt"]: weight, entry["bt"]: bias,
         entry["st"]: scale, entry["sht"]: shift, entry["y"]: y_out},
        entry["ws"],
        handle=handle,
    )
    return y_out


@torch.library.register_fake("cudnn_fused::dw_act_bn")
def _dw_act_bn_fake(input, weight, bias, scale, shift, act_kind):
    return torch.empty_like(input)


class FusedDWActBN(nn.Module):
    """Replacement for Residual's inner Sequential(Conv2d(dw), act_fn, BatchNorm2d)."""

    def __init__(self, conv, act_kind_int, bn):
        super().__init__()
        C = conv.in_channels
        K = conv.kernel_size[0]
        self.act_kind = act_kind_int
        w = conv.weight.detach().clone().to(memory_format=torch.channels_last)
        b = conv.bias.detach().clone().view(1, C, 1, 1) if conv.bias is not None else torch.zeros(1, C, 1, 1)
        with torch.no_grad():
            bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            bn_shift = bn.bias - bn.running_mean * bn_scale
        self.register_buffer("weight", w.half())
        self.register_buffer("bias", b.half().to(conv.weight.device))
        self.register_buffer("scale", bn_scale.view(1, C, 1, 1).half())
        self.register_buffer("shift", bn_shift.view(1, C, 1, 1).half())

    def forward(self, x):
        return torch.ops.cudnn_fused.dw_act_bn(x, self.weight, self.bias, self.scale, self.shift, self.act_kind)


def fuse_convmixer_dw_blocks(model, act_kind_int, verbose=True):
    """model.model is the nn.Sequential; DW blocks are the '0' child of each block-Sequential
    (a Residual wrapping Sequential(Conv2d, act_fn, BatchNorm2d))."""
    n_fused = 0
    for block in model.model:
        if isinstance(block, nn.Sequential):
            for i, sub in enumerate(block):
                # sub is Residual(nn.Sequential(Conv2d, act, BN)) for the DW sub-block
                if sub.__class__.__name__ == "Residual" and isinstance(sub.fn, nn.Sequential):
                    inner = sub.fn
                    if len(inner) == 3 and isinstance(inner[0], nn.Conv2d) and isinstance(inner[2], nn.BatchNorm2d):
                        conv, bn = inner[0], inner[2]
                        if conv.groups == conv.in_channels == conv.out_channels:
                            sub.fn = FusedDWActBN(conv, act_kind_int, bn)
                            n_fused += 1
    if verbose:
        print(f"Fused {n_fused} DW+act+BN block(s) into cudnn_fused::dw_act_bn")
    return model


def build_model(model_name, act_kind_int, device):
    from convmixer import load_pretrained
    m = load_pretrained(model_name, checkpoints_dir=os.path.join(_REPO, "convmixer", "checkpoints"),
                        device=device, channels_last=True)
    m = m.eval()
    m_ref = copy.deepcopy(m).half().to(memory_format=torch.channels_last)
    m_fused = fuse_convmixer_dw_blocks(m.half().to(memory_format=torch.channels_last), act_kind_int)
    return m_ref, m_fused


def check_correctness(m_ref, m_fused, x):
    with torch.no_grad():
        y_ref = m_ref(x)
        y_fused = m_fused(x)
    diff = (y_ref.float() - y_fused.float()).abs()
    rel = diff.max().item() / (y_ref.float().abs().mean().item() + 1e-6)
    print(f"  correctness: max abs diff={diff.max().item():.5f}  ref mean abs={y_ref.float().abs().mean().item():.5f}  rel={rel:.5f}")
    return rel


def time_eager(model, x):
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED):
            model(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED * 1000


def time_reduce_overhead(model, x):
    compiled = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        for _ in range(10):
            compiled(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED):
            compiled(x)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED * 1000


MODELS = [
    ("768_32", 1),   # relu
    ("1536_20", 0),  # gelu
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import csv
    rows = []
    for model_name, act_kind_int in MODELS:
        print(f"=== {model_name} (act_kind={'relu' if act_kind_int else 'gelu'}) ===", flush=True)
        m_ref, m_fused = build_model(model_name, act_kind_int, device)
        x = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
        rel = check_correctness(m_ref, m_fused, x)

        eager_ref_us = time_eager(m_ref, x)
        eager_fused_us = time_eager(m_fused, x)
        print(f"  eager: ref(ATen/cuDNN unfused)={eager_ref_us:.2f}us  fused={eager_fused_us:.2f}us")

        ro_error = ""
        ro_fused_us = None
        try:
            ro_fused_us = time_reduce_overhead(m_fused, x)
            print(f"  reduce-overhead fused: {ro_fused_us:.2f}us")
        except Exception as e:
            ro_error = str(e)
            print(f"  reduce-overhead FAILED: {e}")

        rows.append(dict(
            model=model_name, correctness_rel_err=f"{rel:.5f}",
            eager_ref_us=f"{eager_ref_us:.3f}", eager_fused_us=f"{eager_fused_us:.3f}",
            ro_fused_us=f"{ro_fused_us:.3f}" if ro_fused_us else "",
            ro_error=ro_error,
        ))
        del m_ref, m_fused
        torch.cuda.empty_cache()

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "correctness_rel_err", "eager_ref_us", "eager_fused_us", "ro_fused_us", "ro_error",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
