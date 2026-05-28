# inteligencia-computacional-proyecto

Sistema de lectura y clasificación de placas vehiculares colombianas.

- **Detección** de la placa con YOLO11n (modelo entrenado propio).
- **Clasificación de tipo** por reglas: color HSV (amarilla = particular/moto, blanca = público) + formato vía OCR (`ABC123` = carro, `ABC12D` = moto).
- Pensado para una app **React Native** que corre la inferencia en el dispositivo.

## Estructura

- `train.py` — entrenamiento del detector (Ultralytics YOLO11n).
- `data.yaml` — configuración del dataset (1 clase: `placa`).
- `classify_prototype.py` — prototipo/validación de la lógica de clasificación de tipo.
