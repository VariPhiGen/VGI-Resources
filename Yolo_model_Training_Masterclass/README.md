# 🦾 YOLO Model Training Masterclass (PPE Detection)

End-to-end guide for training a custom YOLO model (PPE detection) using Ultralytics, exporting it, and integrating basic experiment tracking with MLflow.

---

## 📺 Video Tutorial

[![YOLO Model Training Masterclass](https://img.youtube.com/vi/4ogj72UqUxc/maxresdefault.jpg)](https://www.youtube.com/watch?v=4ogj72UqUxc)

**Full walkthrough:** [YouTube Video](https://www.youtube.com/watch?v=4ogj72UqUxc)

---

## 🔎 Overview

This masterclass shows how to:

- Train **YOLOv8** on a custom **PPE (helmet/vest) dataset**
- Configure the **`data.yaml`** file and dataset structure
- Run `model.train()` with key hyperparameters
- Test the trained model on sample images
- **Export** the trained model to **ONNX**
- Set up basic **MLflow** tracking for experiments

All steps are demonstrated in the notebook `Variphi_yolo_training.ipynb`.

---

## 📦 Prerequisites

- **Python 3.8+**
- Basic knowledge of:
  - Object detection concepts
  - YOLO / Ultralytics workflow
  - Python & Jupyter/Colab

Required packages (from the notebook):

```bash
pip install ultralytics opencv-python Pillow mlflow
```

---

## 📁 Project Structure

```text
Yolo_model_Training_Masterclass/
│
├── Variphi_yolo_training.ipynb      # Main training & export notebook
└── Object_detection_video/
    ├── raw_testing_file.ipynb       # Inference on video
    ├── clean_production_code.py     # Production-style inference script
    ├── best.pt                      # Trained YOLO weights
    └── construction-site-video.mp4  # Example test video
```

> The PPE dataset (images, labels, `data.yaml`) is unzipped inside the runtime (e.g. `/content/`).

---

## 🧠 Notebook Flow (Variphi_yolo_training.ipynb)

### 1️⃣ Setup & Installation

- Install Ultralytics and Pillow:

```python
!pip install ultralytics
!pip install Pillow

from ultralytics import YOLO
from PIL import Image
```

- Unzip the **PPE dataset** (images, labels, `data.yaml`):

```python
!unzip "/content/ppe_dataset.zip"
# This creates: train/, val/, test/, data.yaml, README files
```

### 2️⃣ Load Pretrained YOLO Model

Start from a small YOLOv8 model and fine-tune it:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # YOLOv8 nano
```

### 3️⃣ Train on Custom PPE Dataset

Train the model using the Roboflow PPE dataset (`data.yaml` points to train/val/test):

```python
results = model.train(
    data="/content/data.yaml",  # dataset config
    epochs=10,
    imgsz=640
)
```

This produces a run folder like:

```text
/content/runs/detect/train2/
  ├── weights/
  │   ├── best.pt   # best model (used later)
  │   └── last.pt
  └── metrics, plots, logs...
```

### 4️⃣ Test the Trained Model

Run inference on a test image from the dataset:

```python
results = model.predict(
    "/content/test/images/Aitin2331_jpg.rf.f0517ee8bb0bc9613f5747c7e561eb20.jpg"
)

result = results[0]
boxes = result.boxes.xyxy   # bounding boxes
img_with_boxes = result.plot()
Image.fromarray(img_with_boxes)
```

You’ll see detections like **Helmet** and **Vest** with bounding boxes and confidence scores.

### 5️⃣ Export to ONNX

Export the best trained weights to ONNX for deployment:

```python
model.export(format="onnx")
```

This generates something like:

```text
/content/runs/detect/train2/weights/best.onnx
```

You can inspect it with tools like Netron and load it in other runtimes (C++, .NET, etc.).

### 6️⃣ Quick Training Checklist (Summary Cell)

The notebook summarizes the full pipeline:

```text
label dataset → zip → GPU accessible → upload → imports →
update the yaml file location → model.train() → (best.pt, last.pt) →
export → download → perform inference
```

This is the **high-level recipe** you can reuse for any YOLO detection project.

### 7️⃣ MLflow Integration (Basics)

The notebook also shows how to enable **MLflow** tracking:

```python
!pip install mlflow
from ultralytics import settings

# Enable MLflow inside Ultralytics
settings.update({"mlflow": True})
settings.reset()  # optional: reset to defaults if needed

# Example env vars
!export MLFLOW_EXPERIMENT_NAME="Variphi-testing"
!export MLFLOW_RUN="Variphi-testing-running"

# Start MLflow server (for local tracking)
!mlflow server --backend-store-uri runs/mlflow
```

Use this if you want to log experiments, metrics, and artifacts over multiple runs.

---

## 🚀 How to Use This Project

1. **Open the notebook** `Variphi_yolo_training.ipynb` in Jupyter / Colab.  
2. **Upload your dataset zip** (if running in Colab) and update the path in the unzip + `data.yaml` steps.  
3. **Run cells in order**: install deps → unzip dataset → load model → train → test → export.  
4. **Download `best.pt` or `best.onnx`** for deployment.  
5. (Optional) Enable **MLflow** if you want experiment tracking.

---

## 🔍 Key Takeaways

- YOLO training only needs **3 core things**: a **dataset**, a **`data.yaml`**, and a **base model**.  
- Use **`model.train()`** for training, **`model.predict()`** for quick tests, and **`model.export()`** for deployment formats (ONNX, etc.).  
- The same pipeline works for **any custom object detection problem** – just swap the dataset and labels.

---

**Happy Training! 🚀**

*Use this notebook + video to quickly go from raw labeled data to a deployable YOLO model.*
