#!/usr/bin/env python3
"""
Tile-size and channel-blocking sensitivity sweep — 31×31 depthwise, 128ch, 56×56.

For each (M_TILE, CB) combination:
  - Compiles a CUDA kernel variant via torch.utils.cpp_extension.load_inline
  - Benchmarks latency with cold L2 cache flush between runs
  - Reports: latency (ms), throughput, shmem per block (bytes), grid shape, blocks launched

Usage:
    python benchmark_sensitivity.py --device cuda:0 --num_runs 500 \
        --output results/ada_sensitivity_31x31.csv

Notes:
  - CB values {32, 64, 128} are used (96 does not divide 128 evenly).
    Original plan requested {32, 64, 96}; 96 is substituted with 128.
  - Kernel is a "naive" variant (weights in shmem, input from L2/global),
    structurally equivalent to the production kernel but unoptimized for throughput.
    This isolates tile-size effects cleanly.
  - Cold L2 flush: a 160 MB dummy buffer is touched before each timing block to
    evict the 96 MB Ada L2 cache (RTX 6000 Ada).
"""

import torch
import math
import argparse
import csv
import time
import os
import sys

# ─── CUDA kernel template ─────────────────────────────────────────────────────
# Parameters substituted at compile time: M_TILE_VAL, CB_VAL

KERNEL_TEMPLATE = r"""
#include <cuda_fp16.h>
#include <torch/extension.h>

#define K_SIZE  31
#define H_SIZE  56
#define W_SIZE  56
#define C_SIZE  128
#define PAD     15
#define M_TILE  {M}
#define CB      {Cb}
#define NTHREADS 512

// Weight shmem: K*K*CB half values
// Grid: (ceil(56/M), ceil(56/M), 128/CB)
// Each thread (linearly indexed within block) handles one or more
//   (pixel_within_tile, channel_within_cb) assignments.
__global__ void depthwise_31x31_sens_m{M}_cb{Cb}(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output)
{{
    extern __shared__ __half shmem[];

    const int tid     = threadIdx.x;
    const int blockCh = blockIdx.z * CB;
    const int blockI  = blockIdx.x * M_TILE;
    const int blockJ  = blockIdx.y * M_TILE;

    // Load weights: K_SIZE*K_SIZE*CB half values
    const int n_wt = K_SIZE * K_SIZE * CB;
    for (int i = tid; i < n_wt; i += NTHREADS) {{
        int kh     = i / (K_SIZE * CB);
        int rem    = i % (K_SIZE * CB);
        int kw     = rem / CB;
        int c_loc  = rem % CB;
        shmem[i]   = weight[(kh * K_SIZE + kw) * C_SIZE + blockCh + c_loc];
    }}
    __syncthreads();

    // Assign (pixel, channel) pairs to threads; loop if tile > NTHREADS
    const int total = M_TILE * M_TILE * CB;
    for (int idx = tid; idx < total; idx += NTHREADS) {{
        int pix    = idx / CB;
        int c_loc  = idx % CB;
        int pi     = pix / M_TILE;
        int pj     = pix % M_TILE;
        int oi     = blockI + pi;
        int oj     = blockJ + pj;
        if (oi >= H_SIZE || oj >= W_SIZE) continue;
        int ch = blockCh + c_loc;

        __half acc = __float2half(0.0f);
        for (int ki = 0; ki < K_SIZE; ++ki) {{
            int ii = oi + ki - PAD;
            for (int kj = 0; kj < K_SIZE; ++kj) {{
                int ij = oj + kj - PAD;
                if (ii >= 0 && ii < H_SIZE && ij >= 0 && ij < W_SIZE) {{
                    acc = __hfma(
                        input[(ii * W_SIZE + ij) * C_SIZE + ch],
                        shmem[(ki * K_SIZE + kj) * CB + c_loc],
                        acc);
                }}
            }}
        }}
        output[(oi * W_SIZE + oj) * C_SIZE + ch] = acc;
    }}
}}

// L2 flush kernel: touch a large buffer to evict Ada's 96 MB L2
__global__ void l2_flush(const float* __restrict__ buf, float* __restrict__ out, int N) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = buf[i] + 1.0f;
}}

torch::Tensor run_depthwise_31x31_sens_m{M}_cb{Cb}(
    torch::Tensor input, torch::Tensor weight)
{{
    auto output = torch::zeros_like(input);
    int gx = ({H} + M_TILE - 1) / M_TILE;
    int gy = ({W} + M_TILE - 1) / M_TILE;
    int gz = C_SIZE / CB;
    dim3 grid(gx, gy, gz);
    dim3 block(NTHREADS);
    size_t shmem_bytes = (size_t)K_SIZE * K_SIZE * CB * sizeof(__half);

    // Allow up to 164 KB dynamic shmem (Ada supports this)
    cudaFuncSetAttribute(
        depthwise_31x31_sens_m{M}_cb{Cb},
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem_bytes);

    depthwise_31x31_sens_m{M}_cb{Cb}<<<grid, block, shmem_bytes>>>(
        (__half*)input.data_ptr<at::Half>(),
        (__half*)weight.data_ptr<at::Half>(),
        (__half*)output.data_ptr<at::Half>());
    return output;
}}

torch::Tensor run_l2_flush(torch::Tensor buf) {{
    int N = buf.numel();
    auto out = torch::empty_like(buf);
    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;
    l2_flush<<<blocks, tpb>>>(buf.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("run_sens", &run_depthwise_31x31_sens_m{M}_cb{Cb},
          "31x31 sensitivity kernel M={M} CB={Cb}");
    m.def("run_l2_flush", &run_l2_flush, "L2 cache flush");
}}
"""


_SM_SHMEM_LIMIT = {
    # (sm_major, sm_minor) → max dynamic shmem bytes per block
    # SM 7.x: 48KB; SM 8.0: 164KB; SM 8.6/8.9: 99KB; SM 9.0: 228KB; SM 12.x: 99KB
    (7, 0): 49152, (7, 5): 49152,
    (8, 0): 167936, (8, 6): 101376, (8, 9): 101376,
    (9, 0): 232448,
    (12, 0): 101376, (12, 1): 101376,
}
_DEFAULT_SHMEM_LIMIT = 49152


def get_shmem_limit(device):
    """Return the max dynamic shmem per block for the given device."""
    major = torch.cuda.get_device_capability(device)[0]
    minor = torch.cuda.get_device_capability(device)[1]
    return _SM_SHMEM_LIMIT.get((major, minor), _DEFAULT_SHMEM_LIMIT)


def compile_variant(M, Cb, device, cache_dir=None):
    """Compile and return a (module, grid_info, shmem_bytes) for the given (M, Cb).

    Returns None if shmem exceeds the hardware limit for this device.
    """
    from torch.utils.cpp_extension import load_inline

    shmem_bytes = 31 * 31 * Cb * 2  # sizeof(half) = 2
    limit = get_shmem_limit(device)
    if shmem_bytes > limit:
        return None, None, shmem_bytes

    major, minor = torch.cuda.get_device_capability(device)
    arch_flag = f"--gpu-architecture=sm_{major}{minor}"

    src = KERNEL_TEMPLATE.format(M=M, Cb=Cb, H=56, W=56)
    name = f"depthwise_sens_m{M}_cb{Cb}"

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"),
                                 ".cache", "torch_extensions", "sensitivity")
    os.makedirs(cache_dir, exist_ok=True)

    mod = load_inline(
        name=name,
        cpp_sources=[""],
        cuda_sources=[src],
        extra_cuda_cflags=["-O3", "--use_fast_math", arch_flag],
        verbose=False,
    )

    gx = math.ceil(56 / M)
    gy = math.ceil(56 / M)
    gz = 128 // Cb
    blocks_total = gx * gy * gz

    return mod, (gx, gy, gz, blocks_total), shmem_bytes


def benchmark_variant(mod, input_t, weight_t, flush_buf, device,
                      num_runs=500, warmup=50):
    """Benchmark the compiled variant, return latency_ms."""
    flush_out = torch.empty(flush_buf.shape, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            mod.run_sens(input_t, weight_t)
        torch.cuda.synchronize(device)

        # Timed runs with cold L2 flush before each call
        start = time.perf_counter()
        for _ in range(num_runs):
            # Touch the 160 MB buffer to flush L2
            flush_out = mod.run_l2_flush(flush_buf)
            torch.cuda.synchronize(device)
            mod.run_sens(input_t, weight_t)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    # elapsed = num_runs × (flush + kernel). Subtract flush overhead separately.
    # Actually: we want kernel-only latency. Measure flush overhead separately.
    start2 = time.perf_counter()
    for _ in range(num_runs):
        flush_out = mod.run_l2_flush(flush_buf)
        torch.cuda.synchronize(device)
    torch.cuda.synchronize(device)
    flush_overhead = (time.perf_counter() - start2) / num_runs * 1000  # ms

    total_per_call = elapsed / num_runs * 1000  # ms
    kernel_latency = total_per_call - flush_overhead
    return kernel_latency


def check_correctness(mod, device):
    """Spot-check correctness against a reference computation."""
    torch.manual_seed(42)
    H, W, C, K = 56, 56, 128, 31
    pad = 15

    input_nhwc = torch.randn(1, H, W, C, dtype=torch.float16, device=device)
    weight_kkc  = torch.randn(K, K, C, dtype=torch.float16, device=device)

    out_custom = mod.run_sens(input_nhwc.contiguous(), weight_kkc.contiguous())

    # Reference: use PyTorch conv2d (NCHW)
    input_nchw  = input_nhwc.permute(0, 3, 1, 2).contiguous().float()
    weight_nchw = weight_kkc.permute(2, 0, 1).unsqueeze(1).float()  # [C,1,K,K]
    out_ref = torch.nn.functional.conv2d(input_nchw, weight_nchw,
                                         padding=pad, groups=C)
    out_ref_nhwc = out_ref.permute(0, 2, 3, 1).half()

    max_diff = (out_custom - out_ref_nhwc).abs().max().item()
    return max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output", default="results/ada_sensitivity_31x31.csv")
    parser.add_argument("--skip_correctness", action="store_true")
    parser.add_argument("--M_values", nargs="+", type=int, default=[4, 8, 16],
                        help="Output tile sizes to sweep")
    parser.add_argument("--Cb_values", nargs="+", type=int, default=[32, 64, 128],
                        help="Channel block sizes to sweep (must divide 128)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 70)
    print("Tile Sensitivity Sweep — 31×31 depthwise, 128ch, 56×56")
    print("=" * 70)
    print(f"Device:   {torch.cuda.get_device_name(device)}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Runs:     {args.num_runs} (warmup={args.warmup})")
    print(f"M values: {args.M_values}")
    print(f"Cb values: {args.Cb_values}")
    print()

    # Validate Cb values divide 128
    for Cb in args.Cb_values:
        if 128 % Cb != 0:
            print(f"ERROR: Cb={Cb} does not divide C=128 evenly. "
                  f"Use values from {{1, 2, 4, 8, 16, 32, 64, 128}}.")
            sys.exit(1)

    H, W, C = 56, 56, 128
    input_t  = torch.randn(H, W, C, dtype=torch.float16, device=device).contiguous()
    weight_t = torch.randn(31, 31, C, dtype=torch.float16, device=device).contiguous()

    # L2 flush buffer: 160 MB to flush Ada's 96 MB L2
    flush_size = 160 * 1024 * 1024 // 4  # float32 elements
    flush_buf  = torch.randn(flush_size, dtype=torch.float32, device=device)

    results = []

    for M in args.M_values:
        for Cb in args.Cb_values:
            shmem_bytes_req = 31 * 31 * Cb * 2
            shmem_limit = get_shmem_limit(device)
            print(f"\n  M={M:2d}, Cb={Cb:3d}  (shmem={shmem_bytes_req//1024}KB)  — ", end="", flush=True)

            if shmem_bytes_req > shmem_limit:
                print(f"SKIP (exceeds SM shmem limit {shmem_limit//1024}KB)")
                results.append({
                    "M": M, "Cb": Cb,
                    "latency_ms": "SKIP_SHMEM",
                    "shmem_bytes": shmem_bytes_req,
                    "shmem_kb": round(shmem_bytes_req / 1024, 1),
                    "grid": "N/A", "blocks": "N/A",
                    "max_diff": f"shmem>{shmem_limit//1024}KB",
                })
                continue

            print("compiling...", end="", flush=True)
            try:
                mod, (gx, gy, gz, blocks), shmem = compile_variant(M, Cb, device)
            except Exception as e:
                print(f"  COMPILE FAILED: {e}")
                results.append({
                    "M": M, "Cb": Cb,
                    "latency_ms": "FAIL",
                    "shmem_bytes": shmem_bytes_req,
                    "shmem_kb": round(shmem_bytes_req / 1024, 1),
                    "grid": "?", "blocks": "?", "max_diff": "?",
                })
                continue

            shmem_kb = shmem / 1024

            if not args.skip_correctness:
                max_diff = check_correctness(mod, device)
                corr_str = f"max_diff={max_diff:.4f}"
                ok = max_diff < 3.0  # 31*31=961 FP16 MACs; empirical tolerance ~1.5
            else:
                max_diff = -1
                corr_str = "skipped"
                ok = True

            if not ok:
                print(f"  CORRECTNESS FAIL (max_diff={max_diff:.4f}) — skipping benchmark")
                results.append({
                    "M": M, "Cb": Cb,
                    "latency_ms": "CORR_FAIL",
                    "shmem_bytes": shmem,
                    "grid": f"({gx},{gy},{gz})",
                    "blocks": blocks,
                    "max_diff": max_diff,
                })
                continue

            print(f"ok ({corr_str})  shmem={shmem_kb:.1f}KB  grid=({gx},{gy},{gz})={blocks}blk  benchmarking...", end="", flush=True)

            latency = benchmark_variant(mod, input_t, weight_t, flush_buf, device,
                                        num_runs=args.num_runs, warmup=args.warmup)
            print(f"  {latency:.3f} ms")

            results.append({
                "M": M, "Cb": Cb,
                "latency_ms": round(latency, 4),
                "shmem_bytes": shmem,
                "shmem_kb": round(shmem_kb, 1),
                "grid": f"({gx},{gy},{gz})",
                "blocks": blocks,
                "max_diff": round(max_diff, 4) if max_diff >= 0 else "skipped",
            })

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY — 31×31 Tile Sensitivity (cold L2, ms per call)")
    print("=" * 70)
    print(f"{'M':>4} {'Cb':>4} {'Latency':>10} {'Shmem':>9} {'Grid':>14} {'Blocks':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r['M']:>4} {r['Cb']:>4} {str(r['latency_ms']):>10} "
              f"{str(r.get('shmem_kb','?'))+'KB':>9} {r['grid']:>14} {r['blocks']:>7}")

    # Write CSV
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        fieldnames = ["M", "Cb", "latency_ms", "shmem_bytes", "shmem_kb", "grid", "blocks", "max_diff"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to: {args.output}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
