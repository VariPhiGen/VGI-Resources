# 🎯 Object Detection Masterclass - YOLO Implementation Guide

A comprehensive tutorial on YOLO (You Only Look Once) object detection using Ultralytics. Learn how to detect objects in images, understand bounding box formats, and manually plot detection results.

---

## 📺 Video Tutorial

[![Object Detection Masterclass Tutorial](https://img.youtube.com/vi/F9FoXq5l_HY/maxresdefault.jpg)](https://www.youtube.com/watch?v=F9FoXq5l_HY)

**Watch the complete explanation:** [YouTube Video](https://www.youtube.com/watch?v=F9FoXq5l_HY)

---

## 📋 Table of Contents

- [Overview](#overview)
- [What You'll Learn](#what-youll-learn)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Notebook Contents](#notebook-contents)
- [Bounding Box Formats Explained](#bounding-box-formats-explained)
- [Code Examples](#code-examples)
- [How to Use](#how-to-use)

---

## 🎯 Overview

This project demonstrates YOLO object detection using Ultralytics, covering:

- **Loading YOLO models** (YOLO11n)
- **Making predictions** on images
- **Understanding bounding box formats** (xywh, xywhn, xyxy, xyxyn)
- **Manual bounding box plotting** with OpenCV
- **Rescaling coordinates** when needed
- **Visualizing detection results**

---

## 🎓 What You'll Learn

✅ Load and use YOLO models for object detection  
✅ Understand different bounding box coordinate formats  
✅ Know when to rescale coordinates (normalized vs pixel)  
✅ Manually draw bounding boxes on images  
✅ Extract detection information (classes, confidences, boxes)  
✅ Visualize detection results effectively  

---

## 📦 Prerequisites

- **Python 3.8+** installed
- Basic understanding of **Python programming**
- Familiarity with **NumPy** and **OpenCV** (helpful but not required)

---

## 🚀 Installation

### Step 1: Clone or Download this Repository

```bash
git clone <repository-url>
cd VGI_Resources/Object_Detection_Masterclass
```

### Step 2: Install Required Packages

```bash
pip install ultralytics opencv-python matplotlib pillow numpy
```

**Package Descriptions:**
- **ultralytics:** YOLO models and utilities
- **opencv-python:** Image processing and drawing
- **matplotlib:** Visualization
- **pillow:** Image handling
- **numpy:** Array operations

---

## 📁 Project Structure

```
Object_Detection_Masterclass/
│
├── Object_detection_Image/
│   ├── raw_testing_file.ipynb    # Main tutorial notebook
│   ├── clean_production_code.py   # Production-ready code
│   ├── yolo11n.pt                # YOLO model weights
│   └── bus.jpg                   # Sample test image
└── README.md                      # This file
```

---

## 📚 Notebook Contents

### 1. **Model Loading**

Load a pre-trained YOLO model:

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # Load YOLO11 nano model
```

### 2. **Making Predictions**

Run inference on images:

```python
results = model("https://ultralytics.com/images/bus.jpg")
```

### 3. **Understanding Results**

Access detection information:
- **Boxes:** Bounding box coordinates
- **Classes:** Detected object classes
- **Confidences:** Detection confidence scores
- **Names:** Class name mappings

### 4. **Bounding Box Formats**

Learn about four different coordinate formats:
- **xywh:** Center coordinates + width/height (pixels)
- **xywhn:** Normalized center coordinates (0-1)
- **xyxy:** Corner coordinates (pixels)
- **xyxyn:** Normalized corner coordinates (0-1)

### 5. **Manual Plotting**

Draw bounding boxes manually using:
- OpenCV for drawing rectangles
- Coordinate conversion (center to corner format)
- Rescaling normalized coordinates

---

## 🔑 Bounding Box Formats Explained

### Format Comparison

| Format | Description | Rescaling Needed? | Use Case |
|--------|-------------|-------------------|----------|
| **xywh** | Center (x,y) + width, height in pixels | ❌ No | Direct pixel coordinates |
| **xywhn** | Normalized center (0-1) + width, height | ✅ Yes | Model training, normalized data |
| **xyxy** | Top-left (x1,y1) + bottom-right (x2,y2) in pixels | ❌ No | Easiest for manual plotting |
| **xyxyn** | Normalized corner coordinates (0-1) | ✅ Yes | Normalized corner format |

### Coordinate System

```
(0,0) ──────────────────────────► X-axis
  │     
  │    (x1,y1)┌─────────┐
  │           │         │
  │           │  OBJECT │
  │           │         │
  │           └─────────┘(x2,y2)
  │
  ▼
Y-axis
```

**Center Format (xywh):**
- `center_x, center_y`: Center point of bounding box
- `width, height`: Dimensions of the box

**Corner Format (xyxy):**
- `x1, y1`: Top-left corner
- `x2, y2`: Bottom-right corner

### When to Rescale?

**✅ NO Rescaling Needed:**
- `boxes.xywh` - Already in pixel coordinates
- `boxes.xyxy` - Already in pixel coordinates

**⚠️ Rescaling Required:**
- `boxes.xywhn` - Multiply by image width/height
- `boxes.xyxyn` - Multiply by image width/height

---

## 💻 Code Examples

### Basic Object Detection

```python
from ultralytics import YOLO
from PIL import Image

# Load model
model = YOLO("yolo11n.pt")

# Predict
results = model("image.jpg")

# Visualize
result = results[0]
image = result.plot()
Image.fromarray(image)
```

### Manual Plotting with xywh (No Rescaling)

```python
import cv2
import numpy as np

for result in results:
    img = result.orig_img.copy()
    boxes_xywh = result.boxes.xywh.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    
    for i, box in enumerate(boxes_xywh):
        center_x, center_y, width, height = box
        
        # Convert center to corner format
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"{result.names[int(classes[i])]} {confidences[i]:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
```

### Manual Plotting with xywhn (With Rescaling)

```python
for result in results:
    img = result.orig_img.copy()
    img_height, img_width = img.shape[:2]
    
    boxes_xywhn = result.boxes.xywhn.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    
    for i, box in enumerate(boxes_xywhn):
        center_x_norm, center_y_norm, width_norm, height_norm = box
        
        # ⚠️ RESCALE to pixel coordinates
        center_x = center_x_norm * img_width
        center_y = center_y_norm * img_height
        width = width_norm * img_width
        height = height_norm * img_height
        
        # Convert to corner format
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
```

### Using xyxy (Easiest Method)

```python
for result in results:
    img = result.orig_img.copy()
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.astype(int)
        
        # Direct plotting - no conversion needed!
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        label = f"{result.names[int(classes[i])]} {confidences[i]:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
```

### Visualizing Results

```python
import matplotlib.pyplot as plt

for result in results:
    img_with_boxes = result.plot()  # Built-in plotting method
    plt.figure(figsize=(10, 8))
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.title("Object Detection Results")
    plt.show()
```

---

## 🔍 Key Takeaways

1. **Use xyxy format** for easiest manual plotting (corner coordinates)
2. **xywh requires conversion** from center to corner format
3. **Normalized formats (xywhn, xyxyn)** need rescaling by image dimensions
4. **Pixel formats (xywh, xyxy)** are ready to use directly
5. **Built-in `result.plot()`** is the quickest way to visualize
6. **Access detection data** via `result.boxes`, `result.cls`, `result.conf`

---

## 💡 Tips & Best Practices

### Choosing the Right Format

- **For Manual Plotting:** Use `boxes.xyxy` (easiest, no conversion)
- **For Training/ML:** Use normalized formats (`xywhn`, `xyxyn`)
- **For Direct Coordinates:** Use pixel formats (`xywh`, `xyxy`)

### Performance Tips

- Use `.cpu().numpy()` to convert tensors to NumPy arrays
- Cache model loading if processing multiple images
- Batch processing for multiple images

### Common Pitfalls

1. **Forgetting to rescale** normalized coordinates
2. **Mixing coordinate systems** (center vs corner)
3. **Not converting tensors** to NumPy before plotting
4. **Wrong color format** (BGR vs RGB for OpenCV)

---

## 🚀 How to Use

### Option 1: Jupyter Notebook

```bash
jupyter notebook Object_detection_Image/raw_testing_file.ipynb
```

### Option 2: Google Colab

1. Upload the notebook to Google Colab
2. Install ultralytics: `!pip install ultralytics`
3. Run cells sequentially

### Option 3: Python Script

Use the `clean_production_code.py` file for production implementations.

---

## 📖 Additional Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO GitHub Repository](https://github.com/ultralytics/ultralytics)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Object Detection Guide](https://docs.ultralytics.com/modes/predict/)

---

## 📝 Important Notes

- **Model Download:** YOLO models are automatically downloaded on first use
- **Image Formats:** Supports common formats (jpg, png, etc.)
- **GPU Support:** Automatically uses GPU if available
- **Confidence Threshold:** Adjustable in model parameters

---

## 🎯 Summary

This tutorial covers:

✅ YOLO model loading and inference  
✅ Understanding bounding box coordinate formats  
✅ Manual bounding box plotting  
✅ Coordinate conversion and rescaling  
✅ Visualizing detection results  

**Perfect for:** Developers learning object detection, computer vision enthusiasts, and anyone working with YOLO models.

---

**Happy Learning! 🚀**

*Master object detection with YOLO and understand the fundamentals of bounding box coordinates!*

