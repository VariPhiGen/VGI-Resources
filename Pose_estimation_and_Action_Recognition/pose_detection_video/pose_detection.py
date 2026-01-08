from ultralytics import YOLO
import cv2


MODEL_PATH = "/Users/aryan/Desktop/LMS_Experiment/pose_detection_video/best_pose.pt"
DEFAULT_VIDEO = "/Users/aryan/Desktop/LMS_Experiment/pose_detection_video/construction-site-video.mp4"


def run_video(video_path: str = DEFAULT_VIDEO, frame_stride: int = 5) -> None:
    """
    Run YOLO pose detection on the given video path.
    frame_stride controls how often to run inference to save compute.
    """
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {video_path}")
        return

    print("Stream opened successfully! Starting YOLO detection...")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame; stream ended or connection lost")
            break

        count += 1

        if count % frame_stride == 0:
            result = model(frame)
            annotated_frame = result[0].plot()
            cv2.imshow("Pose Detection", annotated_frame)
            print(f"Processed frame {count}")

        # Wait for 'q' key to quit, else wait 1 ms
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit signal received; stopping.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream processing stopped")


if __name__ == "__main__":
    run_video()