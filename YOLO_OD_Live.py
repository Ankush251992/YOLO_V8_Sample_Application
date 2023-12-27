

from ultralytics import YOLO
import cv2
import numpy as np
import time  # Add this line

# Load YOLOv8 model
print("[INFO] loading YOLOv8 from disk...")
model = YOLO("yolov8n.pt")

# Initialize the video stream from the webcam
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)  # Allow camera sensor to warm up

# Loop over frames from the video stream
while True:
    # Read the next frame from the stream
    (grabbed, frame) = vs.read()

    # If the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # Run YOLOv8 prediction on the frame
    results = model([frame])[0]

    # Process results
    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy().squeeze()
        x1, y1, x2, y2 = map(int, [xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
        conf = float(box.conf)
        cls = int(box.cls)
        if conf > 0.5:  # Confidence threshold
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{results.names[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.release()