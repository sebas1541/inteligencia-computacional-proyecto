# Detector y Clasificador de Placas Vehiculares Colombianas

Sistema de **detección y clasificación de placas vehiculares colombianas** basado
en visión por computador, pensado para correr la inferencia **en el dispositivo**
(on-device) dentro de la app móvil **QuickPlate** (React Native / Expo).

El pipeline tiene tres etapas:

1. **Detección de la placa** en la imagen con una red **YOLO11n** entrenada por
   nosotros (1 clase: `placa`).
2. **Lectura de caracteres (OCR)** del recorte de la placa, on-device
   (Apple Vision en iOS, ML Kit en Android).
3. **Clasificación del tipo de vehículo** mediante **reglas** (color de fondo de
   la placa + formato del texto leído), no una segunda red neuronal.

> **Por qué reglas y no una segunda red:** el tipo de placa en Colombia está
> totalmente determinado por dos señales triviales de extraer una vez tienes la
> placa localizada y leída (su **color** y su **formato**). Entrenar un segundo
> clasificador sería redundante, más pesado y menos interpretable. Esto se validó
> experimentalmente (ver §5).

---

## Tabla de contenido

- [1. Tecnologías](#1-tecnologías)
- [2. Dataset](#2-dataset)
- [3. Modelo y algoritmo (YOLO11n)](#3-modelo-y-algoritmo-yolo11n)
- [4. Entrenamiento](#4-entrenamiento)
- [5. Clasificación del tipo de vehículo](#5-clasificación-del-tipo-de-vehículo)
- [6. Guard del OCR (6 caracteres)](#6-guard-del-ocr-6-caracteres)
- [7. Exportación a TFLite](#7-exportación-a-tflite)
- [8. Integración con la app](#8-integración-con-la-app)
- [9. Estructura del repositorio](#9-estructura-del-repositorio)
- [10. Reproducir el entrenamiento](#10-reproducir-el-entrenamiento)
- [11. Cita del dataset](#11-cita-del-dataset)
- [12. Autor y licencia](#12-autor-y-licencia)

---

## 1. Tecnologías

| Capa | Tecnología | Versión / nota |
|------|-----------|----------------|
| Framework de detección | **Ultralytics YOLO** | YOLO11n (modelo *nano*) |
| Backend de entrenamiento | **PyTorch** | acelerado con **MPS** (Apple Silicon) |
| Hardware de entrenamiento | **Mac mini M4** | 10‑core CPU · 10‑core GPU |
| Lenguaje | **Python** | 3.11 |
| Visión clásica (clasificación) | **OpenCV** | umbrales HSV de color |
| Exportación a móvil | **TensorFlow Lite** vía `onnx2tf` / `ai-edge-litert` | fp16 y fp32 |
| OCR on-device | **Apple Vision** (iOS) · **ML Kit** (Android) | en el repo de la app |
| Anotación / dataset | **Roboflow Universe** | export formato YOLOv11 |

Dependencia principal (`requirements.txt`): `ultralytics` — que arrastra
`torch`, `opencv-python`, `numpy` y `pyyaml`.

---

## 2. Dataset

- **Fuente:** *Placas Colombia Dataset*, alojado en **Roboflow Universe**
  (ver [cita completa](#11-cita-del-dataset)).
- **Tarea:** detección de objetos, **1 clase** → `placa`.
- **Formato de export:** **YOLOv11** (layout de Ultralytics: `images/` + `labels/`
  con cajas normalizadas `cx cy w h`).
  - *Nota:* primero se exportó por error como **COCO** (JSON), que Ultralytics no
    lee directamente; se reexportó como YOLOv11.
- **Tamaño total:** **1.106 imágenes**, repartidas en:

  | Split | Imágenes |
  |-------|---------:|
  | `train` | 900 |
  | `valid` | 104 |
  | `test`  | 102 |
  | **Total** | **1.106** |

La **única fuente de verdad** de dónde vive el dataset es la clave `path` en
[`data.yaml`](data.yaml); tanto `train.py` como los scripts de validación lo leen
desde ahí, así que para reentrenar con otro dataset solo se edita esa ruta.

```yaml
# data.yaml
path: /ruta/al/dataset/Placas Colombia.yolov11
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['placa']
```

---

## 3. Modelo y algoritmo (YOLO11n)

**YOLO** (*You Only Look Once*) es un detector de objetos de **una sola pasada**
(*single-shot*): predice cajas y confianza directamente sobre una rejilla de la
imagen, sin propuesta de regiones, lo que lo hace muy rápido — ideal para
**inferencia en tiempo real en un teléfono**.

- Variante: **YOLO11n** (*nano*), la más liviana de la familia YOLO11, elegida por
  el presupuesto de cómputo de un móvil.
- **Entrada:** imagen `416 × 416 × 3` (RGB, normalizada a `0..1`).
- **Salida cruda:** tensor `[1, 5, 3549]`:
  - filas `0..3` → `cx, cy, w, h` (caja, normalizada `0..1` sobre la entrada 416),
  - fila `4` → confianza de la clase única `placa` (ya con sigmoide aplicada),
  - `3549` = número de anclas (combinación de las rejillas multiescala de YOLO).
- **Post-proceso** (en la app, `decode.ts`): se decodifica el tensor y se aplica
  **NMS** (*Non-Maximum Suppression*) con `conf = 0.4` e `IoU = 0.45` para quedarse
  con la mejor caja por placa.

---

## 4. Entrenamiento

El entrenamiento se lanza con [`train.py`](train.py) usando transfer learning
desde los pesos preentrenados `yolo11n.pt`.

**Hardware:** se entrenó en una **Mac mini M4 (10‑core CPU / 10‑core GPU)**,
usando la GPU integrada vía el backend **MPS** de PyTorch (`device="mps"`).

### Hiperparámetros (de `runs/placa_detector/args.yaml`)

| Parámetro | Valor |
|-----------|-------|
| Modelo base | `yolo11n.pt` (preentrenado en COCO) |
| `epochs` | **100** (tope) |
| `patience` | **20** (early-stopping) |
| `imgsz` | **416** |
| `batch` | **16** |
| `device` | **mps** (Apple Silicon; usar `cpu` si no hay GPU) |
| `optimizer` | `auto` (Ultralytics elige; típicamente AdamW/SGD) |
| `lr0` / `lrf` | `0.01` / `0.01` |
| `momentum` / `weight_decay` | `0.937` / `0.0005` |
| `warmup_epochs` | `3` |
| Augmentations | mosaic, fliplr 0.5, HSV (h .015 / s .7 / v .4), translate .1, scale .5, erasing .4, RandAugment |
| `seed` | `0` (determinista) |

### Cuántas épocas tardó

Aunque el tope era **100 épocas**, el entrenamiento se detuvo solo por
**early-stopping** (`patience = 20`) tras **70 épocas corridas**. La **mejor época**
fue la **época 60** (la usada para el modelo final exportado).

### Métricas finales (validación)

Medidas sobre el split `valid` durante el entrenamiento
(`runs/placa_detector/results.csv`):

| Métrica | Valor (≈, mejor época 60) |
|---------|--------------------------:|
| **Precision** | **0.989** |
| **Recall** | **1.000** |
| **mAP@50** | **0.994** |
| **mAP@50‑95** | **0.859** |

> Recall = 1.0 y mAP@50 ≈ 0.99 indican que el detector localiza prácticamente
> todas las placas del set de validación; el mAP@50‑95 (~0.86) refleja la calidad
> del ajuste fino de la caja a umbrales de IoU más estrictos. Las curvas e
> imágenes de batches están en `runs/placa_detector/` (`results.csv`, `labels.jpg`,
> `train_batch*.jpg`).

---

## 5. Clasificación del tipo de vehículo

El tipo se infiere con **reglas** sobre el recorte de la placa
([`classify_prototype.py`](classify_prototype.py) es el prototipo validado; la
versión de producción vive en `decode.ts` / `ocr.ts` de la app):

### Señal 1 — Color de fondo (HSV con OpenCV)

| Color de placa | Significado |
|----------------|-------------|
| **Amarilla** | Vehículo **particular** (carro o moto) |
| **Blanca** | Transporte **público** |

```python
# Umbrales (OpenCV HSV: H 0-179, S 0-255, V 0-255)
YELLOW_LO, YELLOW_HI = (15, 70, 70), (45, 255, 255)
WHITE_LO,  WHITE_HI  = (0, 0, 170), (179, 40, 255)
COLOR_MIN_FRAC = 0.15   # fracción mínima de píxeles del color para aceptarlo
```

### Señal 2 — Formato del texto (OCR)

El formato de las placas colombianas desambigua **carro vs. moto** dentro de las
particulares (amarillas):

| Formato | Patrón | Tipo |
|---------|--------|------|
| `ABC123` | 3 letras + 3 números (`LLLNNN`) | **Carro** |
| `ABC12D` | 3 letras + 2 números + 1 letra (`LLLNNL`) | **Moto** |

### Regla final

- **Amarilla** + formato moto → **moto**; amarilla sin formato de moto → **particular** (carro).
- **Blanca** → **público**.
- En otro caso → **desconocido**.

> **Por qué NO se usa el *aspect ratio* de la caja:** se descartó tras validar —
> la relación de aspecto de la caja no separa moto de carro de forma fiable.
> En la app, además, la detección genérica de vehículo (moto/carro de COCO) sirve
> como verificación cruzada extra para reforzar la señal de "moto".

Colores de acento usados en toda la UI (fuente única en `decode.ts`):
**carro/particular = morado `#4F46E5`**, **moto = rojo `#EF4444`**,
**público = ámbar `#F59E0B`**.

---

## 6. Guard del OCR (6 caracteres)

Una placa colombiana **siempre tiene exactamente 6 caracteres**. El OCR on-device
puede devolver ruido (líneas extra, símbolos, confusiones tipo `0/O`), así que la
app aplica un **guard estricto** (`forntend-inteligencia/src/lib/plate-detect/ocr.ts`):
**solo se acepta una placa si alguna ventana de 6 caracteres cumple uno de los dos
formatos válidos.** Si nada cumple, no se muestra placa (`plate = null`).

Cómo funciona:

1. **Limpieza:** se unen las líneas del OCR (una moto puede venir en 2 renglones),
   se pasa a mayúsculas y se quitan los caracteres no alfanuméricos.
2. **Ventana deslizante de 6:** se recorre cada candidato probando subcadenas de
   exactamente 6 chars.
3. **Coacción por posición:** cada posición se fuerza a su clase esperada
   corrigiendo confusiones típicas del OCR:
   - dígito→letra: `0→O, 1→I, 2→Z, 4→A, 5→S, 6→G, 8→B`
   - letra→dígito: `O/Q/D→0, I/L→1, Z→2, A→4, S→5, B→8, G→6, T→7`
4. **Decisión carro/moto:** se mira el **último carácter** natural — si es letra se
   intenta primero `moto` (`LLLNNL`), si es dígito se intenta primero `carro`
   (`LLLNNN`).

Resultado: lecturas robustas y, sobre todo, **cero falsos positivos** de cadenas
que no son placas (un texto cualquiera no encaja en `LLLNNN`/`LLLNNL`).

---

## 7. Exportación a TFLite

Para correr en el móvil, el modelo entrenado (`best.pt`) se exporta a **TFLite**
con [`export_tflite.py`](export_tflite.py):

```python
model = YOLO("runs/placa_detector/weights/best.pt")
model.export(format="tflite", imgsz=416)
```

Internamente Ultralytics pasa por **ONNX → SavedModel (onnx2tf) → TFLite**. Se
generan dos variantes (en `models/`):

| Archivo | Tamaño | Uso |
|---------|-------:|-----|
| `placa_detector.pt` | 20 MB | pesos PyTorch (entrenamiento / referencia) |
| `placa_detector_fp16.tflite` | 5.1 MB | **app** — mitad de precisión, más liviano |
| `placa_detector_fp32.tflite` | 10 MB | fallback CPU (precisión completa) |

`calibration_image_sample_data_20x128x128x3_float32.npy` es el set de calibración
que usa el exportador.

---

## 8. Integración con la app

El modelo no corre solo aquí; se consume desde la app **QuickPlate**
(`forntend-inteligencia`):

- **`react-native-fast-tflite`** carga el `.tflite` y ejecuta `model.run([buffer])`.
- **`@shopify/react-native-skia`** decodifica el frame de cámara, lo recorta/escala
  a `416×416` y arma el `Float32` RGB de entrada.
- **`decode.ts`** decodifica el tensor `[1,5,3549]`, aplica NMS y clasifica el tipo.
- **`modules/plate-ocr`** (módulo nativo Expo): OCR on-device (Vision / ML Kit).
- **`ocr.ts`** aplica el [guard de 6 caracteres](#6-guard-del-ocr-6-caracteres).

Todo corre **localmente en el teléfono** — no se envía la imagen a ningún servidor.

---

## 9. Estructura del repositorio

```
.
├── README.md                         · este documento
├── SESSION-LOG.md                    · bitácora detallada de toda la construcción
├── data.yaml                         · config del dataset (path + splits + clases)
├── requirements.txt                  · dependencias (ultralytics)
├── train.py                          · entrenamiento del detector (YOLO11n)
├── export_tflite.py                  · export best.pt → TFLite (fp16/fp32)
├── classify_prototype.py             · prototipo de clasificación de tipo (color+formato)
├── detect_preview.py / .jpg          · script + muestra de detección sobre imágenes
├── calibration_image_sample_data_*.npy · datos de calibración para el export
├── models/                           · modelos listos para usar
│   ├── placa_detector.pt             · pesos PyTorch
│   ├── placa_detector_fp16.tflite    · TFLite fp16 (el que usa la app)
│   └── placa_detector_fp32.tflite    · TFLite fp32 (fallback CPU)
└── runs/placa_detector/              · salida del entrenamiento
    ├── args.yaml                     · todos los hiperparámetros usados
    ├── results.csv                   · métricas por época
    ├── labels.jpg / train_batch*.jpg · visualizaciones
    └── weights/                      · best.pt, last.pt, best.onnx, best_tflite/
```

*(El directorio `venv-export/` es el entorno virtual de exportación y está fuera
del control de versiones.)*

---

## 10. Reproducir el entrenamiento

```bash
# 1. Entorno
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Descargar el dataset de Roboflow (formato YOLOv11) y apuntar data.yaml
#    editando la clave `path:` a la carpeta descargada.

# 3. Entrenar (YOLO11n, 100 épocas con early-stopping)
python train.py
#    -> salida en runs/placa_detector/ (pesos en weights/best.pt)

# 4. Exportar a TFLite para la app
python export_tflite.py
#    -> runs/placa_detector/weights/best_tflite/best_float16.tflite
```

---

## 11. Cita del dataset

```bibtex
@misc{ placas-colombia-ixdpr_dataset,
  title = { Placas Colombia Dataset },
  type = { Open Source Dataset },
  author = { usco },
  howpublished = { \url{ https://universe.roboflow.com/usco-thj9e/placas-colombia-ixdpr } },
  url = { https://universe.roboflow.com/usco-thj9e/placas-colombia-ixdpr },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2025 },
  month = { dec },
  note = { visited on 2026-06-04 },
}
```

> *Placas Colombia Computer Vision Model* — publicado en Roboflow Universe.
> Agradecimiento al autor del dataset (`usco`) por liberarlo como Open Source.

---

## 12. Autor y licencia

- **Autor principal:** Sebastián Cañón Castellanos (cód. 202127352) —
  Universidad Pedagógica y Tecnológica de Colombia (UPTC), Tunja.
- **Coautores:**
  - Kevin Johann Jiménez Poveda (cód. 202220120) — UPTC, Tunja.
  - Pedro Eduardo Cruz López (cód. 202128778) — UPTC, Tunja.
- **Curso:** Inteligencia Computacional — UPTC.
- **Repos relacionados:**
  - `inteligencia-computacional-proyecto` — este repo (ML).
  - `inteligencia-backend` — API FastAPI (cuentas + historial de placas).
  - `forntend-inteligencia` — app móvil **QuickPlate** (Expo / React Native).
- **Modelo base:** YOLO11n (Ultralytics, licencia AGPL-3.0).
- **Dataset:** *Placas Colombia* (Roboflow Universe) — ver cita arriba.

Para la bitácora técnica completa (decisiones, errores y soluciones de toda la
sesión) ver [`SESSION-LOG.md`](SESSION-LOG.md).
