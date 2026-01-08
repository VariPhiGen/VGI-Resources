import os
from ultralytics import settings, YOLO

# Configure MLflow to use local server
os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'

# Set experiment and run names (optional but recommended)
os.environ['MLFLOW_EXPERIMENT_NAME'] = 'ppe_detection_mlflow_Version_2'
os.environ['MLFLOW_RUN'] = 'yolov8_testing_Version_2'

settings.update({"mlflow": True})

# Verify MLflow is enabled and show tracking URI
print(f"MLflow enabled: {settings['mlflow']}")
print(f"MLflow tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")

model = YOLO("/Users/aryan/Desktop/LMS_Experiment/best.pt")



print("Training model...")



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

print("Training completed!")
print("Check your results at: http://127.0.0.1:5000")
