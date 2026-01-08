import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("/Users/aryan/Desktop/LMS_Experiment/Computer_Vision_Solution/Testing_video_Solutions/Retail_Store.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
#video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

from_email = "aryanagrahari215@gmail.com"  # the sender email address
password = "xxxx xxxx xxxx xxxx"  # 16-digits password generated via: https://myaccount.google.com/apppasswords
to_email = "aryanagrahari215@gmail.com"  # the receiver email address

# Initialize security alarm object
securityalarm = solutions.SecurityAlarm(
    show=True,  # display the output
    model="yolo11n.pt",  # i.e. yolo11s.pt, yolo11m.pt
    records=3,  # total detections count to send an email
    classes=[0]
)

securityalarm.authenticate(from_email, password, to_email)  # authenticate the email server

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = securityalarm(im0)

    # print(results)  # access the output

    # video_writer.write(results.plot_im)  # write the processed frame.
    cv2.imshow("Security Alarm", results.plot_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows