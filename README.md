# 🚗 Blindrive – Self-Driving Car (UPBGE + CNN)

## Overview

Blindrive is a self-driving car project built using **UPBGE simulation** and a **CNN regression model**. The system learns to predict steering angles directly from images.

> **🚧 Progress**
> - Model predictions have been improved
> - Now capable of handling **sharp turns more effectively**

> **⚠️ Current Status**
> - Works well on **simple and curved roads**
> - Struggles when **another part of the road is seen**
> - Pure **image-to-steering prediction** (no real physics modeling)
> - Car moves with **slow and mostly constant speed**
> - Still under active improvement

---

## 🧠 Pipeline

```
Simulation (UPBGE)  →  Data Collection  →  Model Training (CNN)  →  Auto Driving + Telemetry
```

---

## Step 1 — Setup Simulation 🎮

Open the Blender project file:

```
simulation/track.blend
```

> **Important:** Use **camera view** instead of rendering. It's faster and produces consistent screenshots for training.

---

## Step 2 — Data Collection 📸

**Script:** `scripts/collect_data.py`

Attach this script as a **Python Component** to the car, then run the simulation and drive manually using the keyboard. Images and steering values will be saved automatically.

**Output structure:**

```
data/
├── images/
└── labels.csv
```

---

## Step 3 — Train the Model 🧠

**Script:** `scripts/train_model.py`

The CNN model predicts steering angles based on an **NVIDIA-style architecture**.

**Run:**

```bash
python scripts/train_model.py
```

**Output:**

```
model/
└── model_best.onnx
```

---

## Step 4 — Auto Driving + Telemetry Logging 🤖

**Script:** `scripts/ai_drive_and_TelemetryLogger.py`

Attach the script to the car in UPBGE, then press **P** to start the simulation. The car will drive itself using the trained model.

**What it logs:**

| Data | Description |
|------|-------------|
| Steering | Predicted angle at each frame |
| Position | X and Y coordinates |
| Rotation | Z-axis rotation |

**Output:**

```
output/telemetry/
└── manual_drive.csv
```

---

## Demo Videos 📊

Two road samples were used for testing:

| Test | Description | Link |
|------|-------------|------|
| 1 | Simple road | [Watch](https://youtu.be/Ojz3Fqyc_M8) |
| 2 | Curvy road | [Watch](https://youtu.be/GlgoxTPY9U4) |
| 3 | Curvy road with camera rotation | [Watch](https://youtu.be/tkHCuT7yrfs) |
---

## ⚠️ Limitations

- Model struggles on **sharp turns**
- No recovery behaviour — going off-road results in failure
- Only works with standard road colours

---

## 📂 Dataset

The dataset is **not included** (size exceeds 2 GB). Use the collection script to generate your own.

**Format:**

```
images/
└── frame_000001.png

labels.csv:
image, steering
frame_000001.png, 0.25
```

**To generate:**

```bash
python scripts/collect_data.py
```

---

## 🚀 Future Improvements

- Better performance on sharp curves
- Recovery driving behaviour when off-road
- Smarter camera input (not relying on full screenshots)
- Combined image + position-based learning

---

## 🛠 Requirements

```
tensorflow
numpy
opencv-python
onnxruntime
```

---

## 📝 Notes

- Uses **camera view instead of rendering** for speed and consistency
- Model expects consistent preprocessing — training and inference pipelines must match exactly

---

## Summary

Blindrive is an experimental self-driving system that collects its own driving data, trains a CNN model, and drives autonomously inside a simulation. It performs well under simple conditions and serves as a solid foundation for further development.
