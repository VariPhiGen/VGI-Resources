import os
from ultralytics import settings, YOLO
import wandb

# Login to Weights & Biases
wandb.login(key="6e959256cb799c2d5ee8a329be3dffae615531dc")

# Enable W&B logging for Ultralytics
settings.update({"wandb": True})

# Verify W&B is enabled
print(f"W&B enabled: {settings['wandb']}")

# Load the YOLO model
model = YOLO("/Users/aryan/Desktop/LMS_Experiment/best.pt")

print("Training model...")

# Train the model with W&B logging
results = model.train(
    data="/Users/aryan/Desktop/LMS_Experiment/ppe_dataset/data.yaml",
    epochs=2,
    imgsz=640,
    batch=16,
    patience=50,
    save=True,
    plots=True,
    device="mps",
    project="ultralytics_demo",  # W&B project name
    name="yolov8_testing_Version_3",  # W&B run name
)

print("Training completed!")
print("Check your results in the Weights & Biases dashboard!")

