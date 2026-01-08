import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("/Users/aryan/Desktop/LMS_Experiment/Computer_Vision_Solution/Testing_video_Solutions/Retail_Store.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
#video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# For object counting with heatmap, you can pass region points.
# region_points = [(20, 400), (1080, 400)]                                      # line points
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]              # rectangle region
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon points


region_points = [(145, 18), (382, 155), (59, 318), (3, 60)]
#region_points = [(286, 192), (64, 414), (4, 167), (27, 63)]

# Initialize heatmap object
heatmap = solutions.Heatmap(
    show=True,  # display the output
    model="yolo11n.pt",  # path to the YOLO11 model file
    colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
    region=region_points,  # object counting with heatmaps, you can pass region_points
    classes=[0],  # generate heatmap for specific classes i.e person and car.
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = heatmap(im0)

    # print(results)  # access the output

    # video_writer.write(results.plot_im)  # write the processed frame.
    cv2.imshow("Heatmap", results.plot_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows