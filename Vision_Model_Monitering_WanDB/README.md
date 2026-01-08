# 📊 Vision Model Monitoring with MLflow & Weights & Biases

Short, practical examples for monitoring YOLO training runs using **MLflow** and **Weights & Biases (W&B)**.

---

## 📺 Video Tutorial

[![Vision Model Monitoring Masterclass](https://img.youtube.com/vi/XFHYaPN55IQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=XFHYaPN55IQ)

**Full walkthrough:** [YouTube Video](https://www.youtube.com/watch?v=XFHYaPN55IQ)

---

## 🔎 Overview

This folder shows **two ways** to monitor YOLO training for a PPE detection model:

- `MLFlow_Monitering.py` → log runs to a local **MLflow** server
- `wandb_monitering.py` → log runs to **Weights & Biases** dashboards

Both scripts reuse the same YOLO model and dataset, and only change **where** metrics and artifacts are logged.

---

## 📦 Prerequisites

- Python 3.8+
- Trained or base YOLO model at:
  - `/Users/aryan/Desktop/LMS_Experiment/best.pt`
- PPE dataset config:
  - `/Users/aryan/Desktop/LMS_Experiment/ppe_dataset/data.yaml`

Install dependencies:

```bash
pip install ultralytics mlflow wandb opencv-python pillow
```

> Make sure your paths to `best.pt` and `data.yaml` are correct on your machine.

---

## 📁 Files

- `MLFlow_Monitering.py`  
  Enable MLflow tracking and train YOLO with metrics logged to a local MLflow server.

- `wandb_monitering.py`  
  Enable W&B logging and train YOLO with rich dashboards in the Weights & Biases UI.

---

## 🧪 1. MLflow Monitoring (`MLFlow_Monitering.py`)

### What it does

- Points Ultralytics to a **local MLflow tracking server** (`http://127.0.0.1:5000`)
- Enables MLflow inside Ultralytics settings
- Trains YOLO on the PPE dataset
- Logs metrics, parameters, and artifacts (plots, weights) to MLflow

### Key code (conceptually)

```python
from ultralytics import settings, YOLO
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'ppe_detection_mlflow_Version_2'
os.environ['MLFLOW_RUN'] = 'yolov8_testing_Version_2'

settings.update({"mlflow": True})

model = YOLO("/Users/aryan/Desktop/LMS_Experiment/best.pt")

results = model.train(
    data="/Users/aryan/Desktop/LMS_Experiment/ppe_dataset/data.yaml",
    epochs=3,
    imgsz=640,
    batch=16,
    patience=50,
    save=True,
    plots=True,
    device="mps",
)
```

### How to run

1. **Start MLflow server** (one time, in a separate terminal):

   ```bash
   mlflow server --backend-store-uri runs/mlflow
   ```

2. **Run the script**:

   ```bash
   python MLFlow_Monitering.py
   ```

3. Open MLflow UI in your browser:

   ```text
   http://127.0.0.1:5000
   ```

You’ll see runs under the experiment `ppe_detection_mlflow_Version_2`.

---

## 📈 2. Weights & Biases Monitoring (`wandb_monitering.py`)

### What it does

- Logs into **Weights & Biases** using your API key
- Enables W&B integration in Ultralytics settings
- Trains YOLO with the same PPE dataset
- Sends metrics, curves, system stats, and artifacts to W&B

### Key code (conceptually)

```python
import wandb
from ultralytics import settings, YOLO

wandb.login(key="<YOUR_WANDB_API_KEY>")
settings.update({"wandb": True})

model = YOLO("/Users/aryan/Desktop/LMS_Experiment/best.pt")

results = model.train(
    data="/Users/aryan/Desktop/LMS_Experiment/ppe_dataset/data.yaml",
    epochs=2,
    imgsz=640,
    batch=16,
    patience=50,
    save=True,
    plots=True,
    device="mps",
    project="ultralytics_demo",
    name="yolov8_testing_Version_3",
)
```

> **Important:** Replace the hard-coded API key with your own secure method (env vars or `wandb login`).

### How to run

1. **Login to W&B** (once):

   ```bash
   wandb login
   ```

   or ensure the script has your correct API key.

2. **Run the script**:

   ```bash
   python wandb_monitering.py
   ```

3. Go to your W&B dashboard and open the project `ultralytics_demo` to see the run `yolov8_testing_Version_3`.

---

## 🔍 When to Use Which

- **Use MLflow** if you want:
  - A **self-hosted**, local or on-prem tracking server
  - Simple experiment logging and model registry

- **Use W&B** if you want:
  - Rich visual dashboards, comparison tools, and collaboration
  - Cloud-hosted tracking with minimal setup

Both approaches wrap the **same YOLO training call**, so you can easily switch between them.

---

## 📝 Summary

This folder gives you **ready-to-run examples** of:

- Enabling **MLflow** logging for YOLO training
- Enabling **Weights & Biases** logging for YOLO training
- Keeping your **training code identical**, while changing only the monitoring backend.

Use these scripts as templates to plug monitoring into any **Ultralytics / YOLO** vision project.

---

**Happy Monitoring! 📊🚀**

*Track your experiments like a pro, whether you prefer MLflow or Weights & Biases.*
