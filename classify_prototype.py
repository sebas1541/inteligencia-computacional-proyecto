"""Prototipo de clasificacion de tipo de placa (Camino 1).

Logica (validada contra imagenes reales):
  - COLOR (HSV)  -> amarilla = particular/moto ; blanca = publico
  - FORMATO OCR  -> ABC123 (3 num) = carro ; ABC12D (2 num + letra) = moto

El aspect ratio de la caja NO separa moto de carro (se descarto tras validar).
En la app, ademas, la deteccion de vehiculo (moto/carro de COCO) da una senal
de moto aun mas robusta como cruce de verificacion.
"""
import os
import re
import glob
from pathlib import Path
import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent


def dataset_root():
    """Lee la ubicacion del dataset desde data.yaml (clave `path`)."""
    with open(REPO / "data.yaml") as f:
        return Path(yaml.safe_load(f)["path"])

# Umbrales de color (OpenCV HSV: H 0-179, S 0-255, V 0-255)
YELLOW_LO, YELLOW_HI = (15, 70, 70), (45, 255, 255)
WHITE_LO, WHITE_HI = (0, 0, 170), (179, 40, 255)
COLOR_MIN_FRAC = 0.15

# Formatos de placa colombiana
FORMAT_MOTO = re.compile(r"^[A-Z]{3}\d{2}[A-Z]$")   # ABC12D
FORMAT_CARRO = re.compile(r"^[A-Z]{3}\d{3}$")        # ABC123


def plate_color(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yf = float(cv2.inRange(hsv, YELLOW_LO, YELLOW_HI).mean()) / 255.0
    wf = float(cv2.inRange(hsv, WHITE_LO, WHITE_HI).mean()) / 255.0
    if yf >= wf and yf > COLOR_MIN_FRAC:
        return "amarilla", yf, wf
    if wf > COLOR_MIN_FRAC:
        return "blanca", yf, wf
    return "indeterminado", yf, wf


def format_of(ocr_text):
    if not ocr_text:
        return None
    t = re.sub(r"[^A-Z0-9]", "", ocr_text.upper())
    if FORMAT_MOTO.match(t):
        return "moto"
    if FORMAT_CARRO.match(t):
        return "carro"
    return None


def classify_plate(img_bgr, box_xyxy, ocr_text=None):
    x1, y1, x2, y2 = box_xyxy
    crop = img_bgr[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return "desconocido", {}
    color, yf, wf = plate_color(crop)
    fmt = format_of(ocr_text)

    if color == "amarilla":
        tipo = "moto" if fmt == "moto" else "particular"  # sin OCR -> particular (moto necesita formato)
    elif color == "blanca":
        tipo = "publico"
    else:
        tipo = "desconocido"
    return tipo, {"color": color, "yellow": round(yf, 2), "white": round(wf, 2), "fmt": fmt}


def yolo_to_xyxy(line, w, h):
    _, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
    return (int((cx - bw / 2) * w), int((cy - bh / 2) * h),
            int((cx + bw / 2) * w), int((cy + bh / 2) * h))


COLORS = {"moto": (0, 165, 255), "particular": (0, 255, 255),
          "publico": (255, 255, 255), "desconocido": (0, 0, 255)}


def main():
    root = dataset_root()
    test_images = root / "test" / "images"
    test_labels = root / "test" / "labels"
    out = REPO / "classify_preview.jpg"

    tiles, counts = [], {}
    for p in sorted(glob.glob(os.path.join(str(test_images), "*"))):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = os.path.join(str(test_labels), os.path.splitext(os.path.basename(p))[0] + ".txt")
        if not os.path.exists(lbl):
            continue
        with open(lbl) as f:
            for line in f:
                if not line.strip():
                    continue
                box = yolo_to_xyxy(line, w, h)
                tipo, _ = classify_plate(img, box)  # sin OCR todavia
                counts[tipo] = counts.get(tipo, 0) + 1
                c = COLORS[tipo]
                cv2.rectangle(img, box[:2], box[2:], c, 2)
                cv2.putText(img, tipo, (box[0], max(15, box[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        tiles.append(cv2.resize(img, (256, 256)))

    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * 256, cols * 256, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = t
    cv2.imwrite(str(out), grid)

    print("=== Conteo por tipo (solo color, sin OCR) ===")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"Montage guardado en: {out}")


if __name__ == "__main__":
    main()
