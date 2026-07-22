"""Probe: does cuDNN 9.25 fix the GB10 fusion NVRTC-JIT failure seen under 9.19?

Loads libcudnn 9.25 from a --target pip dir via PYTHONPATH priority (no env change),
then attempts the exact fused graph that fails under 9.19: conv+bias+GELU, C2, NHWC fp16.
"""
import sys
sys.path.insert(0, "/tmp/cudnn925_py")  # nvidia.cudnn resolves to 9.25 from here

import torch
import cudnn

print("frontend:", cudnn.__version__, "backend:", cudnn.backend_version_string())

device = "cuda"
C, K, H, W = 1536, 9, 32, 32
pad = K // 2
x = torch.randn(1, C, H, W, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
w = torch.randn(C, 1, K, K, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
b = torch.randn(1, C, 1, 1, device=device, dtype=torch.float16)

graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    compute_data_type=cudnn.data_type.FLOAT,
    intermediate_data_type=cudnn.data_type.FLOAT,
)
xt = graph.tensor_like(x)
wt = graph.tensor_like(w)
bt = graph.tensor_like(b)
y = graph.conv_fprop(image=xt, weight=wt, padding=[pad, pad], stride=[1, 1], dilation=[1, 1])
y = graph.bias(input=y, bias=bt)
y = graph.gelu(input=y)
y.set_output(True).set_data_type(cudnn.data_type.HALF)
graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B, cudnn.heur_mode.FALLBACK])
graph.check_support()
n = graph.get_execution_plan_count()
print("candidates:", n)
built = []
for i in range(n):
    try:
        graph.build_plan_at_index(i)
        print(f"plan {i}: BUILD OK  name={graph.get_plan_name_at_index(i)}")
        built.append(i)
    except Exception as e:
        print(f"plan {i}: BUILD FAILED: {str(e)[:200]}")

if built:
    ws_size = max(graph.get_workspace_size_plan_at_index(i) for i in built)
    ws = torch.empty(max(ws_size, 1), device=device, dtype=torch.uint8)
    y_out = torch.empty_like(x)
    tensor_map = {xt: x, wt: w, bt: b, y: y_out}
    for i in built:
        try:
            for _ in range(20):
                graph.execute_plan_at_index(tensor_map, ws, i)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(200):
                graph.execute_plan_at_index(tensor_map, ws, i)
            e.record()
            torch.cuda.synchronize()
            print(f"plan {i}: EXEC OK  {s.elapsed_time(e)/200*1000:.2f} us")
        except Exception as ex:
            print(f"plan {i}: EXEC FAILED: {str(ex)[:150]}")
    # correctness spot-check with the fastest working plan
    import torch.nn.functional as F
    ref = F.conv2d(x, w, bias=None, padding=pad, groups=C) + b
    ref = F.gelu(ref.float()).half()
    diff = (ref.float() - y_out.float()).abs().max().item()
    print("max abs diff vs PyTorch reference:", diff)
else:
    print("NO PLAN BUILT — 9.25 does not fix the GB10 fusion failure")
