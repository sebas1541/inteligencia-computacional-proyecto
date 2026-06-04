"""Entrenamiento del detector de placas (Ultralytics YOLO11n).

El dataset se toma de data.yaml (clave `path`), asi que para reentrenar con
otro dataset solo hay que editar esa ruta, no este script.
"""
from pathlib import Path
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(REPO / "data.yaml"),
        epochs=100,
        imgsz=416,
        batch=16,
        device="mps",          # Apple Silicon; usa "cpu" si no tienes GPU
        patience=20,
        project=str(REPO / "runs"),
        name="placa_detector",
        plots=True,
    )
