# 🧩 YOLO Application Solutions & Projects

Collection of end-to-end **YOLO-based computer vision applications**, built on top of a common YOLO model (`yolo11n.pt`). Each script demonstrates a different real-world use case.

---

## 📺 Video Tutorial

[![YOLO Application Solutions Projects](https://img.youtube.com/vi/7rGmMy_gkKE/maxresdefault.jpg)](https://www.youtube.com/watch?v=7rGmMy_gkKE)

**Full walkthrough:** [YouTube Video](https://www.youtube.com/watch?v=7rGmMy_gkKE)

---

## 🔎 Overview

This folder showcases multiple **production-style YOLO applications**, including:

- Object counting (global and region-based)
- Queue and zone monitoring
- Speed and distance estimation
- Heatmap generation
- Security alarms and alerts
- Instance segmentation and object blurring
- Workout and posture monitoring

All solutions live under `Computer_Vision_Solution/` and share a common YOLO model.

---

## 📦 Prerequisites

- Python 3.8+
- YOLO model weights: `Computer_Vision_Solution/yolo11n.pt`
- Required libraries (typical stack):

```bash
pip install ultralytics opencv-python numpy matplotlib pillow
```

Some scripts may also expect a video input (CCTV footage, demo clips, etc.).

---

## 📁 Project Structure

```text
Yolo_Application_Solutions_Projects/
│
└── Computer_Vision_Solution/
    ├── analytics.py
    ├── distance_calculation.py
    ├── heatmap_generation.py
    ├── instance_segmentation.py
    ├── object_bluring.py
    ├── Object_counting_into_Regions.py
    ├── Object_counting.py
    ├── Object_cropping_class_wise.py
    ├── Queue_management.py
    ├── security_alarm_1.py
    ├── security_alarm.py
    ├── speed_estimation.py
    ├── Track_object_in_Zone.py
    ├── workout_monitering.py
    └── yolo11n.pt
```

---

## 🧠 What Each Script Represents (High Level)

- `Object_counting.py`  
  Count all detected objects in the frame (e.g., people, vehicles).

- `Object_counting_into_Regions.py`  
  Count objects **inside defined regions** (e.g., lanes, zones, areas).

- `Queue_management.py`  
  Monitor queues (length, crowding) using region-based counting and tracking.

- `Track_object_in_Zone.py`  
  Track objects entering/leaving specific zones (virtual fences, areas of interest).

- `distance_calculation.py`  
  Estimate distance between objects or to the camera using simple geometry / calibration.

- `speed_estimation.py`  
  Approximate object speed based on movement across frames and known scale.

- `heatmap_generation.py`  
  Generate spatial **heatmaps** from tracked object positions over time.

- `security_alarm.py` / `security_alarm_1.py`  
  Trigger alerts when objects enter restricted zones or violate rules.

- `instance_segmentation.py`  
  Perform instance segmentation (masks) instead of just bounding boxes.

- `object_bluring.py`  
  Blur selected objects (e.g., faces/plates) for privacy.

- `Object_cropping_class_wise.py`  
  Crop and save detected objects by **class** (e.g., only people, only cars).

- `workout_monitering.py`  
  Use detections/pose to roughly monitor workout form or repetitions.

- `analytics.py`  
  Aggregate counts, time-in-zone, or other metrics for reporting.

> Implementation details may vary per script, but all share the same YOLO inference core.

---

## 🚀 How to Use (Generic Pattern)

Most scripts follow this pattern:

1. **Load YOLO model** (path to `yolo11n.pt`).  
2. **Open a video source** (`cv2.VideoCapture` from file or webcam).  
3. **Run YOLO per frame** to get detections.  
4. **Apply business logic** (counting, zones, speed, alarms, etc.).  
5. **Visualize and/or save output** (annotated video, logs, analytics).

Typical entry point:

```bash
cd Yolo_Application_Solutions_Projects/Computer_Vision_Solution
python Object_counting.py        # or any other script
```

Check each script for:
- Input video path / camera index
- Any required configuration (regions, thresholds, class filters, etc.)

---

## 🎯 Key Idea

You train or load **one YOLO model**, then reuse it across many **domain-specific applications**:

- Retail, security, traffic, sports analytics, fitness, etc.
- All powered by the same detection backbone with different post-processing logic.

Use these scripts as templates to build your own **custom YOLO-powered solutions**.

---

**Have fun building real-world CV apps! 🚀**
