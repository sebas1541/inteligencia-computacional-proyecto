from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data="/Users/sebas1541/Desktop/placas-col/data.yaml",
        epochs=100,
        imgsz=416,
        batch=16,
        device="mps",
        patience=20,
        project="/Users/sebas1541/Desktop/placas-col/runs",
        name="placa_detector",
        plots=True,
    )
