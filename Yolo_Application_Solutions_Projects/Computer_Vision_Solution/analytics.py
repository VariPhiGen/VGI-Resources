import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("/Users/aryan/Desktop/LMS_Experiment/Computer_Vision_Solution/Testing_video_Solutions/Indian_highway_lane.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(
    "analytics_output.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (512, 288),  # this is fixed
)

# Initialize analytics object
# analytics_line= solutions.Analytics(
#     show=True,  # display the output
#     analytics_type="bar",  # pass the analytics type, could be "pie", "bar" or "area".
#     model="yolo11n.pt",  # path to the YOLO11 model file
#     classes=[2],  # display analytics for specific detection classes
# )

# analytics_bar = solutions.Analytics(
#     show=True,
#     analytics_type="bar",
#     model="yolo11n.pt",
#     classes=[2],
# )
analytics_pie = solutions.Analytics(
    show=True,
    analytics_type="pie",
    model="yolo11n.pt",
    classes=[0,1,2,3],
)

# Process video
frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        #results_line = analytics_line(im0, frame_count)
        #results_bar = analytics_bar(im0, frame_count)
        results_pie = analytics_pie(im0, frame_count)
        # out.write(results_line.plot_im)
        #out.write(results_bar.plot_im)
        out.write(results_pie.plot_im)
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()  # destroy all opened windows



# import cv2
# from ultralytics import solutions

# cap = cv2.VideoCapture("/Users/aryan/Desktop/LMS_Experiment/testing_video.mp4")
# assert cap.isOpened(), "Error reading video file"

# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# # Output video: wider to fit both charts side by side
# output_width = 1024  # 512 for line + 512 for bar
# output_height = 760   # Default height for charts

# out = cv2.VideoWriter(
#     "analytics_side_by_side.avi",
#     cv2.VideoWriter_fourcc(*"MJPG"),
#     fps,
#     (output_width, output_height)
# )

# # Initialize analytics objects with show=False
# analytics_line = solutions.Analytics(
#     show=False,              # No auto-display
#     analytics_type="line",
#     model="yolo11n.pt",
#     # classes=[0, 2],        # Uncomment to filter classes (e.g., person and car)
# )

# analytics_bar = solutions.Analytics(
#     show=False,
#     analytics_type="bar",
#     model="yolo11n.pt",
#     # classes=[0, 2],
# )

# frame_count = 0

# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         break

#     frame_count += 1

#     # Process and get results (this runs YOLO + updates charts)
#     results_line = analytics_line(im0, frame_count)  # Returns SolutionResults
#     results_bar = analytics_bar(im0, frame_count)    # Returns SolutionResults

#     # Extract the plotted images
#     line_chart = results_line.plot_im    # np.ndarray of the line chart
#     bar_chart  = results_bar.plot_im     # np.ndarray of the bar chart

#     # Resize if needed (optional)
#     line_chart = cv2.resize(line_chart, (512, 512))
#     bar_chart  = cv2.resize(bar_chart, (512, 512))

#     # Combine side by side: [Line | Bar]
#     combined = cv2.hconcat([line_chart, bar_chart])

#     # Optional: Add titles using green color
#     cv2.putText(combined, "Line Chart (Count over Time)", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     cv2.putText(combined, "Bar Chart (Current Count)", (522, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Write to output
#     out.write(combined)

#     # Optional: Live display
#     cv2.imshow("Line + Bar Analytics Side by Side", combined)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# out.release()
# cv2.destroyAllWindows()

