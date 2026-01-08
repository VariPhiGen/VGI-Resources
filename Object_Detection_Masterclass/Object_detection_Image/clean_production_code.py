from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import uuid
from io import BytesIO

app = FastAPI(title="YOLO Object Detection API", version="1.0.0")

# Load YOLO model
model = YOLO("yolo11n.pt")




class DetectionResponse(BaseModel):
    """Response model for detection results"""
    detected_bbox: List[List[float]]
    success: bool
    message: str
    image_path: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str


class DetectionService:
    """Service class for YOLO object detection"""
    
    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model = YOLO(model_path)
    
    def detect_objects(self, image_bytes: bytes) -> DetectionResponse:
        """Detect objects in uploaded image"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Run YOLO detection
            results = self.model(image)

            # Extract result from YOLO
            result = results[0]  # single image inference


            annotated_image_np = result.plot()
            
            # Convert numpy array (BGR) to PIL Image (RGB)
            annotated_image = Image.fromarray(cv2.cvtColor(annotated_image_np, cv2.COLOR_BGR2RGB))

            annotated_image.save("detected_image.jpg")


            detected_bbox = result.boxes.xyxy.cpu().numpy()


            return DetectionResponse(
                detected_bbox=detected_bbox,
                success=True,
                message="Object detection successful",
                image_path="detected_image.jpg"
            )
            
        except Exception as e:
            return DetectionResponse(
                success=False,
                detections=[],
                total_objects=0,
                message=f"Error during detection: {str(e)}"
            )


# Initialize detection service
detection_service = DetectionService()


@app.get("/", response_model=HealthResponse)
async def startup_router():
    """Startup/Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="YOLO Object Detection API is running"
    )


@app.post("/detect", response_model=DetectionResponse)
async def detection_router(file: UploadFile = File(...)):
    """Object detection endpoint - upload image and get detections"""
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            return DetectionResponse(
                success=False,
                detections=[],
                total_objects=0,
                message="Invalid file type. Please upload an image."
            )
        
        # Perform detection
        result = detection_service.detect_objects(image_bytes)

        return result
        
    except Exception as e:
        return DetectionResponse(
            success=False,
            detections=[],
            total_objects=0,
            message=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("clean_production_code:app", host="0.0.0.0", port=8000, reload=True)

