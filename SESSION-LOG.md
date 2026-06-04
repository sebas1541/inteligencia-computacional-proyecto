# QuickPlate / Inteligencia Computacional — Session Log

End-to-end log of everything built this session: the **license-plate detector**
(training + TFLite export), the **FastAPI backend**, and the **QuickPlate**
Expo/React Native app. Heaviest detail on the **model training**.

## Repos
| Repo | Purpose | GitHub |
|------|---------|--------|
| `inteligencia-computacional-proyecto` | ML — YOLO11n plate detector + classifier + TFLite export | `github.com/sebas1541/inteligencia-computacional-proyecto` |
| `inteligencia-backend` | FastAPI API — accounts + plate history (+ Google login) | `github.com/sebas1541/inteligencia-backend` |
| `forntend-inteligencia` | Expo/RN app **QuickPlate** | `github.com/sebas1541/forntend-inteligencia` |

The app's design is replicated from the user's own **`koen-mobile`** project
(indigo + liquid-glass aesthetic).

---

# 1. Model training (full detail)

### Goal
Detect Colombian vehicle license plates in an image and classify the plate
**type**. Detection by a trained YOLO model; type classification by rules
(color + plate format), not a second network.

### Dataset
- **Source:** Roboflow export, project "Placas Colombia", version v1.
- **Format:** exported as **YOLOv11** (Ultralytics layout). *(First exported as
  COCO by mistake — COCO is JSON and Ultralytics doesn't read it directly;
  re-exported as YOLOv11.)*
- **Location on disk:** `/Users/sebas1541/Desktop/Placas Colombia.yolov11`
- **Size / classes:** ~1.1k images, **1 class: `placa`**.
- **Splits:** **train 900 · valid 104 · test 102** (images each with a matching
  `.txt` label).
- `data.yaml` (the project's, single source of truth) points `path:` at the
  dataset and declares `nc: 1`, `names: ['placa']`.

### Model & framework
- **Ultralytics YOLO11n** (the *nano* variant), fine-tuned from the pretrained
  `yolo11n.pt`. ~2.6M params, 6.4 GFLOPs, 182 layers.
- The model head was auto-adapted from COCO's 80 classes to **`nc=1`**.

### Training environment
- **Python 3.9.6** venv in the repo (`venv/`), created fresh — the original
  training environment was gone.
- **ultralytics 8.4.60**, **torch 2.8.0**, **opencv 4.13**.
- **Device: `mps`** (Apple **M4** GPU). `torch.backends.mps.is_available()` → True.
- `requirements.txt` = `ultralytics` (pulls torch/opencv/numpy/pyyaml).

### Training configuration (`train.py`)
```
model   = YOLO("yolo11n.pt")
data    = <repo>/data.yaml      # reads dataset from data.yaml `path`
epochs  = 100
imgsz   = 416
batch   = 16
device  = "mps"                 # Apple Silicon GPU ("cpu" if no GPU)
patience= 20                    # early stop after 20 epochs w/o improvement
project = <repo>/runs
name    = placa_detector
plots   = True
```
- Optimizer **auto-selected → AdamW**, lr0≈0.002, momentum≈0.9.
- Output → `runs/placa_detector/` (weights/best.pt, weights/last.pt, results.csv, plots). `runs/` is git-ignored.

### Training run notes
- Ran in the background on the M4. **~40–45 s/epoch** (incl. validation).
- Benign warnings during data scan:
  - *"corrupt JPEG restored and saved"* on **2** training images (auto-repaired).
  - *"Box and segment counts should be equal … only boxes will be used"* — the
    Roboflow export mixed some polygon/segmentation annotations with boxes;
    Ultralytics correctly drops the polygons and trains on the bounding boxes.
- Image access was fast; `0 dataloader workers` (MPS limitation).

### Metrics over time (validation, 104 images)
| Epoch | Precision | Recall | mAP@50 | mAP@50-95 |
|------:|----------:|-------:|-------:|----------:|
| ~1  | 0.788 | 0.846 | **0.911** | 0.680 |
| ~16 | 0.987 | 1.000 | **0.995** | 0.812 |
| ~32 | 0.990 | 1.000 | 0.995 | 0.844 |
| ~60 (**best**) | **0.989** | **1.000** | **0.992** | **0.859** |
| 70 (stopped)| 0.989 | 1.000 | 0.994 | 0.847 |

- The model essentially **converged by ~epoch 16** (mAP50 ≈ 0.995). The user
  **manually stopped** training at **epoch 70/100** — no loss, since
  `best.pt` is frozen at the best checkpoint.

### Final result — `best.pt` (epoch 60)
- **Precision 0.989 · Recall 1.000 · mAP@50 0.992 · mAP@50-95 0.859.**
- **Recall 1.000** = it detected *every* plate in the validation set.
- Saved at `runs/placa_detector/weights/best.pt` (~20 MB).

### Real-world validation (`detect_preview.py`)
- Ran `best.pt` over the **102 unseen test images** + the color classifier.
- **108 plates detected** (some images have multiple). Montage saved to
  `detect_preview.jpg` (git-ignored). Visual: tight, correct boxes on
  essentially every plate (TBC-314, SOA-727, CSX-784, SRZ-389, GNK-495, …).

### Type classification (`classify_prototype.py`, rule-based)
- **Color (HSV):** yellow ⇒ *particular/moto*; white ⇒ *público*.
- **Plate format (OCR):** `ABC123` (3 digits) ⇒ *carro*; `ABC12D`
  (2 digits + letter) ⇒ *moto*.
- Aspect ratio was tried and **discarded** (didn't separate moto from carro).
- Plate-type palette is shared with the app (`particular`/`publico`/`moto`/`desconocido`).

### Hardware aside
An **RTX 4070** would train this roughly **4–6× faster** than the M4 (CUDA +
tensor-core FP16 vs MPS). For a nano model on 900 images, a chunk of time is
fixed overhead (dataloading/aug/val), so not a full 10× — but materially faster
for heavy iteration.

---

# 2. TFLite export (full detail — this was a saga)

Goal: convert `best.pt` → **TFLite** so the model runs **on-device** in the app.

### The Python 3.9 wall
- Ultralytics' TFLite pipeline is `PyTorch → ONNX → TF → TFLite` and needs
  **`onnx2tf`**, which requires **Python ≥ 3.10**. The training venv was **3.9**.
- **ONNX export succeeded** (`best.onnx`, 10.1 MB) but the ONNX→TF step failed:
  `onnx2tf ≥1.26` is 3.10-only, plus a **protobuf 6.x** conflict
  (`MessageFactory.GetPrototype` removed) and TF not importable mid-run.

### The fix (kept the model, changed the toolchain)
1. `brew install python@3.11`.
2. Throwaway venv `venv-export/` (Python 3.11), installed **`onnx2tf` 1.28.8 +
   `tensorflow` 2.21 + onnx/onnxslim/onnxruntime + `psutil`** (onnx2tf needs it).
   *No torch/ultralytics needed* — we reused the already-exported `best.onnx`.
3. `onnx2tf -i runs/placa_detector/weights/best.onnx -o …/best_tflite`.

### Output (validated)
- **`best_float16.tflite` — 5.1 MB**
- **`best_float32.tflite` — 10 MB**
- Loaded with the TFLite interpreter:
  - **Input:** `[1, 416, 416, 3]` float32 (**NHWC** — mobile layout)
  - **Output:** `[1, 5, 3549]` float32 — YOLO11 raw detections: **4 bbox coords
    + 1 class score** × **3549** candidate boxes. **No embedded NMS** → the app
    must **decode + run NMS** in JS (Phase 3).
- Refreshed the committed artifacts in `models/` (`placa_detector.pt`,
  `placa_detector_fp16.tflite`, `placa_detector_fp32.tflite`) with the retrained
  model and pushed.

> `venv-export/` (Python 3.11 + TensorFlow) is git-ignored and disposable —
> re-create it only if you need to re-export.

---

# 3. ML repo housekeeping
- De-hardcoded all scripts (were pointing at old `~/Desktop/placas-col` paths):
  `train.py`, `detect_preview.py`, `classify_prototype.py`, `export_tflite.py`
  now resolve repo-relative and read the dataset location from `data.yaml`.
- Added `requirements.txt`; ignored generated artifacts (`detect_preview.jpg`,
  onnx2tf calibration `.npy`).

---

# 4. Backend — `inteligencia-backend` (FastAPI)

- **Stack:** FastAPI + Uvicorn, **SQLAlchemy 2.0 + SQLite**, JWT (**PyJWT**) +
  **bcrypt**. Run: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
  (`0.0.0.0` so a phone on the LAN can reach it). Swagger at `/docs`.
- **Endpoints:**
  - `POST /auth/register` (email, password, full_name) → JWT
  - `POST /auth/login` → JWT
  - `POST /auth/google` → verifies a Google **ID token** (google-auth) against
    `GOOGLE_CLIENT_IDS`, finds/creates the user, issues our JWT
  - `GET /auth/me`
  - `GET/POST /plates`, `DELETE /plates/{id}` — per-user **history**
- **Data model:** `User`(id, email, full_name, hashed_password *nullable*,
  google_sub, avatar_url, created_at), `Plate`(id, owner_id, plate, plate_type,
  color, confidence, note, **lat/lng**, created_at).
- **Env:** `SECRET_KEY`, `GOOGLE_CLIENT_IDS` (comma-separated allowed audiences),
  `DATABASE_URL`. `.env` is git-ignored.
- Verified end-to-end with curl (auth flow + 401/409 cases).

---

# 5. App — `forntend-inteligencia` (QuickPlate)

**Stack:** Expo **SDK 56**, expo-router, TypeScript, React 19 / RN 0.85 (new arch).
Design ported from `koen-mobile` in **StyleSheet** (not NativeWind, for
robustness on the bleeding-edge SDK). Liquid glass via **`expo-glass-effect`**.

### Built, in order
1. **Scaffold + design system** — indigo tokens (`#4F46E5`), light/dark,
   `GlassCard`/`GlassButton`, `QuickPlateLogo`, Home + Scanner screens.
2. **Camera** — `react-native-vision-camera`. ⚠️ npm's `latest` was a **broken
   v5.0.11** (no `app.plugin.js`, crashes Node 25); **pinned to v4.7.3**.
   Needs a dev build (not Expo Go).
3. **Glass bottom tab bar (koen pattern):** `NativeTabs` (real iOS-26 liquid
   glass) on iOS + JS `Tabs` on Android. Tabs: **Mapa · Escanear · Historial · Perfil**.
4. **Auth (koen-style):** login presented as a **modal** — `login` (email) →
   `password`, plus a **register wizard** (email → name → password). Auth gate
   auto-presents login when logged out; closing returns to the map.
   JWT in **expo-secure-store**. UI kit `Button`/`Input`/`Avatar`.
5. **Branding:** real **QuickPlate logo** lockup (light/dark, tight-cropped),
   theme-switched. Cute clay **vehicle icons** (car/moto/van) pulled from koen,
   used in the plate-type legend, history list, and map markers.
6. **Map = home (koen-style):** `MapView` + custom **indigo map style** +
   user location + a **marker per saved plate** + a **gorhom bottom sheet**
   listing scans by distance + glass header (avatar pill + "Escanear" pill) +
   floating recenter button. `expo-location` captures coords on save → plate
   appears on the map.
7. **Google Maps on iOS** (the hard one — see below).
8. **Google Sign-In:** `expo-auth-session` `useIdTokenAuthRequest` → ID token →
   `POST /auth/google`. Client IDs via `EXPO_PUBLIC_GOOGLE_*` env;
   `app.config.ts` derives the iOS reversed-client-id URL scheme.

### Google-Maps-on-iOS fix (notable)
- `react-native-maps` **1.27.2** moved Google support into a `react-native-maps/Google`
  **subspec** and **dropped the standalone `react-native-google-maps.podspec`**.
  Expo's prebuild still injects `pod 'react-native-google-maps'` → `pod install`
  failed (`No podspec found`).
- **Solution:** a config plugin **`plugins/with-rnmaps-google-podspec.js`** that
  **regenerates that podspec for 1.27.2** (sources `ios/AirGoogleMaps/**`,
  deps GoogleMaps **9.4.0** + Google-Maps-iOS-Utils 6.1.0) before `pod install`.
  Result: `provider=PROVIDER_GOOGLE` + the indigo tint work **on iOS too**, while
  staying on the RN-0.85-compatible 1.27.2.

### Secrets / config
- **`.env`** (git-ignored) + **`.env.example`** (placeholders) + dynamic
  **`app.config.ts`** that injects from env. Vars:
  - `EXPO_PUBLIC_GOOGLE_MAPS_API_KEY`
  - `EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID`, `_WEB_CLIENT_ID`, `_ANDROID_CLIENT_ID`
- Google client IDs come from the **same Google Cloud project** as the user's
  ghostline project (new OAuth clients for QuickPlate's bundle, not reused).

### Running it
- **Simulator:** `npx expo run:ios` (iPhone 17, iOS 26 → real liquid glass).
- **Physical iPhone:** requires Apple **signing** (Personal Team set in Xcode →
  Signing & Capabilities → Automatically manage signing) + **Developer Mode**
  on the phone (Settings → Privacy & Security → Developer Mode → On → restart).
  Then `npx expo run:ios --device`. Phone + Mac on same Wi-Fi; backend on
  `0.0.0.0:8000`.
- **Rule of thumb:** JS-only change → Metro reload; **new native module →
  rebuild** (`run:ios`). (The `ExpoLocation`/`ExpoCrypto` "native module not
  found" crashes were always "reloaded JS over an old build".)

---

# 6. Current state / what's left

**Done & pushed:** trained detector + fresh TFLite; FastAPI backend (accounts,
history, Google login); QuickPlate app (glass UI, camera, tab bar, auth modal,
logo, vehicle icons, Google-Maps map-home with tint on iOS, Google sign-in).

**In progress:** running on the physical iPhone (signing ✅, enabling Developer
Mode is the last gate).

**Not yet built — Phase 3 (the real ML in the app):**
- Re-enable vision-camera **frame processors** + worklets.
- Run `placa_detector.tflite` on frames via **`react-native-fast-tflite`**.
- **Decode the `[1,5,3549]` output + NMS** in JS; draw live boxes (Skia).
- Classify by color/format (port `classify_prototype.py`); OCR via ML Kit.
- Auto-save detected plates **with geolocation** → they drop straight onto the map.
