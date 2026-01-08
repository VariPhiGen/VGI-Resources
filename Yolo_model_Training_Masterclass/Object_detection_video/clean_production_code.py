from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from ultralytics import YOLO
import cv2
import numpy as np
import subprocess
import os
from pathlib import Path

app = FastAPI(title="YOLO Video Object Detection API", version="1.0.0")

# Load YOLO model
model = YOLO("yolo11n.pt")


class VideoDetectionResponse(BaseModel):
    """Response model for video detection results"""
    detected_bbox: List[List[List[float]]]  # List of frames, each containing list of bboxes
    frames_processed: int
    success: bool
    message: str
    output_video_path: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str


class VideoDetectionService:
    """Service class for YOLO video object detection"""
    
    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model = YOLO(model_path)
        self.output_dir = Path("output_videos")
        self.output_dir.mkdir(exist_ok=True)
    
    def get_youtube_stream_url(self, youtube_url: str) -> Optional[str]:
        """Extract direct stream URL from YouTube video/live stream"""
        try:
            cmd = [
                'yt-dlp',
                '-f', 'best[ext=mp4]/best',
                '-g',
                youtube_url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            stream_url = result.stdout.strip()
            return stream_url
        except subprocess.CalledProcessError as e:
            print(f"Error extracting stream URL: {e}")
            return None
    
    def detect_objects_in_video(
        self, 
        video_url: str, 
        frame_skip: int = 50,
        max_frames: int = 100,
        save_video: bool = True
    ) -> VideoDetectionResponse:
        """Detect objects in video stream"""
        try:
            # Check if it's a YouTube URL
            if "youtube.com" in video_url or "youtu.be" in video_url:
                print(f"Detected YouTube URL, extracting stream URL...")
                stream_url = self.get_youtube_stream_url(video_url)
                if not stream_url:
                    return VideoDetectionResponse(
                        detected_bbox=[],
                        frames_processed=0,
                        success=False,
                        message="Failed to extract YouTube stream URL"
                    )
            else:
                stream_url = video_url
            
            # Open video stream
            cap = cv2.VideoCapture(stream_url)
            
            if not cap.isOpened():
                return VideoDetectionResponse(
                    detected_bbox=[],
                    frames_processed=0,
                    success=False,
                    message="Failed to open video stream"
                )
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer if saving
            output_path = None
            video_writer = None
            if save_video:
                output_path = str(self.output_dir / "detected_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            all_detections = []
            frame_count = 0
            processed_count = 0
            
            print(f"Starting video processing (processing every {frame_skip} frames)...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every nth frame
                if frame_count % frame_skip == 0:
                    # Run YOLO detection
                    results = self.model(frame)
                    result = results[0]
                    
                    # Get annotated frame
                    annotated_frame = result.plot()
                    
                    # Extract bounding boxes
                    bboxes = result.boxes.xyxy.cpu().numpy().tolist()
                    all_detections.append(bboxes)
                    
                    # Save annotated frame to video
                    if video_writer is not None:
                        video_writer.write(annotated_frame)
                    
                    processed_count += 1
                    print(f"Processed frame {frame_count} (detection frame {processed_count})")
                    
                    # Stop if max frames reached
                    if processed_count >= max_frames:
                        break
            
            # Release resources
            cap.release()
            if video_writer is not None:
                video_writer.release()
            
            return VideoDetectionResponse(
                detected_bbox=all_detections,
                frames_processed=processed_count,
                success=True,
                message=f"Video processing successful. Processed {processed_count} frames.",
                output_video_path=output_path
            )
            
        except Exception as e:
            return VideoDetectionResponse(
                detected_bbox=[],
                frames_processed=0,
                success=False,
                message=f"Error during video detection: {str(e)}"
            )


# Initialize detection service
detection_service = VideoDetectionService()


@app.get("/", response_model=HealthResponse)
async def startup_router():
    """Startup/Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="YOLO Video Object Detection API is running"
    )


@app.post("/detect-video", response_model=VideoDetectionResponse)
async def video_detection_router(
    video_url: str = Form(...),
    frame_skip: int = Form(50),
    max_frames: int = Form(100),
    save_video: bool = Form(True)
):
    """
    Video object detection endpoint
    
    Parameters:
    - video_url: URL of the video stream (YouTube URL, RTSP, HTTP, or local file path)
    - frame_skip: Process every nth frame (default: 50)
    - max_frames: Maximum number of frames to process (default: 100)
    - save_video: Whether to save annotated video (default: True)
    """
    try:
        # Perform detection
        result = detection_service.detect_objects_in_video(
            video_url=video_url,
            frame_skip=frame_skip,
            max_frames=max_frames,
            save_video=save_video
        )
        
        return result
        
    except Exception as e:
        return VideoDetectionResponse(
            detected_bbox=[],
            frames_processed=0,
            success=False,
            message=f"Error processing request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("clean_production_code:app", host="0.0.0.0", port=8001, reload=True)
