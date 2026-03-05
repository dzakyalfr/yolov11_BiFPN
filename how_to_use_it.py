from ultralytics import YOLO

# Build fresh
model = YOLO("ultralytics/cfg/models/11/yolo11-bifpn.yaml")

# Train
model.train(data="coco.yaml", epochs=100, imgsz=640)
