import sys, os
sys.path.insert(0, "convmixer")
sys.path.insert(0, ".")
import torch
import benchmark_cudnn_fused_e2e as fe2e

device = "cuda"
m_ref, m_fused = fe2e.build_model("1536_20", 0, device)

x1 = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
x2 = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)

compiled = torch.compile(m_fused, mode="reduce-overhead")
with torch.no_grad():
    y1_compiled = compiled(x1).clone()
    y2_compiled = compiled(x2).clone()
    y1_ref = m_ref(x1)
    y2_ref = m_ref(x2)

print("y1 vs y2 compiled differ (should be large, different inputs):", (y1_compiled - y2_compiled).abs().max().item())
print("y1_compiled vs y1_ref max diff:", (y1_compiled.float() - y1_ref.float()).abs().max().item(),
      "ref mean abs:", y1_ref.float().abs().mean().item())
print("y2_compiled vs y2_ref max diff:", (y2_compiled.float() - y2_ref.float()).abs().max().item(),
      "ref mean abs:", y2_ref.float().abs().mean().item())

with torch.no_grad():
    y1_again = compiled(x1).clone()
print("y1_compiled repeat consistency diff (should be ~0):", (y1_again - y1_compiled).abs().max().item())

# Confirm many-iteration replay still tracks a changing input (catches "graph frozen on first capture" bugs)
with torch.no_grad():
    outs = []
    for i in range(5):
        xi = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
        outs.append(compiled(xi).clone())
diffs = [(outs[i] - outs[i + 1]).abs().max().item() for i in range(4)]
print("consecutive distinct-input output diffs (should all be large/nonzero):", diffs)
