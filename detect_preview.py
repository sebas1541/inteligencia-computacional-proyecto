"""Valida el pipeline completo sobre el test set (imagenes que el modelo NO vio):
detecta la placa con el modelo entrenado (best.pt) + clasifica el tipo por color.
"""
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from classify_prototype import classify_plate, COLORS

MODEL = "/Users/sebas1541/Desktop/placas-col/runs/placa_detector/weights/best.pt"
TEST_IMAGES = "/Users/sebas1541/Desktop/Proyecto Placas.v1-primera-version.yolov8/test/images"
OUT = "/Users/sebas1541/Desktop/placas-col/detect_preview.jpg"

model = YOLO(MODEL)
tiles, detected = [], 0
for p in sorted(glob.glob(os.path.join(TEST_IMAGES, "*"))):
    img = cv2.imread(p)
    if img is None:
        continue
    res = model(img, verbose=False, conf=0.35)[0]
    for b in res.boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = b[:4]
        detected += 1
        tipo, _ = classify_plate(img, (x1, y1, x2, y2))
        c = COLORS.get(tipo, (0, 0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, tipo, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    tiles.append(cv2.resize(img, (256, 256)))

cols = 6
rows = (len(tiles) + cols - 1) // cols
grid = np.zeros((rows * 256, cols * 256, 3), dtype=np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    grid[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = t
cv2.imwrite(OUT, grid)
print(f"Imagenes: {len(tiles)} | placas detectadas: {detected}")
print(f"Montage: {OUT}")
