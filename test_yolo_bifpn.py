"""End-to-end model build + forward pass test for YOLOv11 + BiFPN neck."""
import sys
import torch

sys.path.insert(0, ".")

from ultralytics.nn.tasks import DetectionModel

print("Building yolo11-bifpn.yaml ...")
model = DetectionModel(
    cfg=r"ultralytics/cfg/models/11/yolo11-bifpn.yaml",
    ch=3, nc=80, verbose=True,
)
model.eval()

dummy = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    out = model(dummy)

# out is a tuple during inference: (predictions, None) or just predictions
preds = out[0] if isinstance(out, (tuple, list)) else out
print(f"\nForward pass OK. Output shape: {tuple(preds.shape)}")
# YOLOv11 detect output: (batch, num_anchors, 4+nc)  where anchors = 80*80+40*40+20*20 = 8400
assert preds.shape[0] == 1
assert preds.shape[1] == 4 + 80, f"Expected 84 ch, got {preds.shape[1]}"  # 4 bbox + nc
assert preds.shape[2] == 8400, f"Expected 8400 anchors, got {preds.shape[2]}"  # 80*80+40*40+20*20
print("All assertions passed!")
