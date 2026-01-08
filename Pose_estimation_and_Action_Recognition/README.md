# 🕺 Pose Estimation & Action Recognition (YOLO Pose)

Concise guide to train and use YOLO pose models for action/keypoint detection. Based on the notebook `Variphi_Pose_estimation.ipynb`.

---

## 📺 Video Tutorial

[![Pose Estimation & Action Recognition](https://img.youtube.com/vi/4R2ZgicA5qk/maxresdefault.jpg)](https://www.youtube.com/watch?v=4R2ZgicA5qk)

**Full walkthrough:** [YouTube Video](https://www.youtube.com/watch?v=4R2ZgicA5qk)

---

## 🔎 Overview

- Train a **YOLO pose** model (`yolo11n-pose.pt`) on a custom dataset (`data.yaml`).
- Run inference on images/videos and visualize keypoints & skeletons.
- Package training artifacts for download.
- Includes a ready video inference example in `pose_detection_video/`.

---

## 📦 Prerequisites

- Python 3.8+
- GPU recommended (CUDA/MPS/Colab).
- Dataset with keypoint labels and a `data.yaml` (train/val/test paths + `kpt_shape`).

Install essentials:
```bash
pip install ultralytics opencv-python pillow
```

---

## 📁 Project Structure

```text
Pose_estimation_and_Action_Recognition/
│
├── Variphi_Pose_estimation.ipynb     # Main training/inference notebook
└── pose_detection_video/
    ├── best_pose.pt                  # Trained pose weights
    ├── pose_detection.py             # Video inference example
    └── testing_video.mp4             # Sample video
```

---

## 🚀 Notebook Flow (short)

1) **Setup**
```python
!pip install ultralytics
import ultralytics; ultralytics.checks()
```

2) **Load base pose model**
```python
from ultralytics import YOLO
model = YOLO("yolo11n-pose.pt")
```

3) **Train on your dataset**
```python
results = model.train(data="/content/data.yaml", epochs=10, imgsz=640)
```
- Uses `kpt_shape` from `data.yaml` (e.g., 12 keypoints × 3 dims).
- Outputs run folder under `/content/runs/pose/train/` with `best.pt`.

4) **Package artifacts** (optional)
```python
import shutil
shutil.make_archive('/content/runs', 'zip', '/content/runs')
```

5) **Load trained weights & infer**
```python
pose_model = YOLO("/content/best_pose.pt")
results = pose_model("/content/train/images/sample.jpg")
img = results[0].plot()  # visualize keypoints/skeleton
```

6) **Video inference example**
See `pose_detection_video/pose_detection.py` (uses `best_pose.pt` + `testing_video.mp4`).

---

## 🎯 Key Points

- **Model**: `yolo11n-pose.pt` → fine-tuned on your keypoint dataset.
- **Config**: `data.yaml` must define paths and `kpt_shape` (e.g., `[12,3]`).
- **Outputs**: `best.pt` for deployment; use `results[0].plot()` to visualize.
- **Artifacts**: Zip the `/runs` directory to download training results.

---

## 🛠️ Quick Commands

Train:
```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO("yolo11n-pose.pt")
model.train(data="/content/data.yaml", epochs=10, imgsz=640)
PY
```

Infer on an image:
```bash
python - <<'PY'
from ultralytics import YOLO
pose_model = YOLO("best_pose.pt")
res = pose_model("testing_video.mp4")  # or a single image path
res[0].save(filename="out.jpg")
PY
```

---

**Happy keypointing! 🧍‍♂️🚀**
