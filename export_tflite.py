"""Exporta el detector entrenado (best.pt) a TFLite para la app React Native."""
from pathlib import Path
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent
WEIGHTS = REPO / "runs" / "placa_detector" / "weights" / "best.pt"

model = YOLO(str(WEIGHTS))
path = model.export(format="tflite", imgsz=416)
print("EXPORTADO:", path)
