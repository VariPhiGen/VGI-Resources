import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("/Users/aryan/Desktop/LMS_Experiment/Computer_Vision_Solution/Testing_video_Solutions/Hotel_Parking_Lot.mp4")
assert cap.isOpened(), "Error reading video file"

# Initialize object cropper
cropper = solutions.ObjectCropper(
    show=True,  # display the output
    model="yolo11n.pt",  # model for object cropping, e.g., yolo11x.pt.
    classes=[2],  # crop specific classes such as person and car with the COCO pretrained model.
    # conf=0.5,  # adjust confidence threshold for the objects.
    crop_dir="cropped-detections"  # set the directory name for cropped detections
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = cropper(im0)

    # print(results)  # access the output

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows