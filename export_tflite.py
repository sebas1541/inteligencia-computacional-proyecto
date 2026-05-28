"""Exporta el detector entrenado (best.pt) a TFLite para la app React Native."""
from ultralytics import YOLO

model = YOLO("/Users/sebas1541/Desktop/placas-col/runs/placa_detector/weights/best.pt")
path = model.export(format="tflite", imgsz=416)
print("EXPORTADO:", path)
