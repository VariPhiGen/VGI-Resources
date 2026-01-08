import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("/Users/aryan/Desktop/LMS_Experiment/Computer_Vision_Solution/Testing_video_Solutions/indian_Truck_loading.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("distance_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize distance calculation object
distancecalculator = solutions.DistanceCalculation(
    model="yolo11n.pt",  # path to the YOLO11 model file.
    show=True,  # display the output# Only calculate distances between person (0) and car (2)
    tracker="botsort.yaml",
    classes=[2,3]
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = distancecalculator(im0)

    print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows