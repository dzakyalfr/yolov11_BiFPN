"""Quick shape-correctness smoke test for the adapted BiFPN."""
import sys
import torch

sys.path.insert(0, ".")
from bifpn import BiFPN

# ---- Typical YOLOv11-n channel sizes after backbone ----
in_ch  = (256, 512, 1024)   # P3, P4, P5 backbone output channels
fs     = 128                 # BiFPN inner feature_size
B      = 2                   # batch size
H, W   = 640, 640            # input image resolution

# Backbone output shapes (stride 8 / 16 / 32)
p3 = torch.randn(B, in_ch[0], H // 8,  W // 8)    # (2, 256, 80, 80)
p4 = torch.randn(B, in_ch[1], H // 16, W // 16)   # (2, 512, 40, 40)
p5 = torch.randn(B, in_ch[2], H // 32, W // 32)   # (2, 1024,20, 20)

model = BiFPN(in_channels=in_ch, feature_size=fs, num_layers=2)
model.eval()

with torch.no_grad():
    outs = model([p3, p4, p5])

assert len(outs) == 3, "Expected 3 outputs."

for i, (out, stride) in enumerate(zip(outs, [8, 16, 32])):
    exp = (B, fs, H // stride, W // stride)
    assert out.shape == exp, f"P{i+3}: expected {exp}, got {out.shape}"
    print(f"  P{i+3}: {tuple(out.shape)}  OK")

print("\nAll shape assertions passed!")
