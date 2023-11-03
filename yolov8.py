from ultralytics import YOLO

#loading a pretrained model
model=YOLO('yolov8n.pt')

#results=model(source="children.mp4",show=True,conf=0.4,save=True)
results=model(source=0,show=True,conf=0.4,save=True)