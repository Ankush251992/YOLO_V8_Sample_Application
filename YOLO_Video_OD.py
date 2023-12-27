
from ultralytics import YOLO
import cv2
import argparse
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
args = vars(ap.parse_args())

# load YOLOv8 model
print("[INFO] loading YOLOv8 from disk...")
model = YOLO("yolov8n.pt")

# initialize the video stream and output writer
vs = cv2.VideoCapture(args["input"])
writer = None

# loop over frames from the video file stream
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    # run YOLOv8 prediction on the frame
    results = model([frame])[0]  # results for the first (and only) image in the batch

    # process results
    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy().squeeze()
        x1, y1, x2, y2 = map(int, [xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
        conf = float(box.conf)
        cls = int(box.cls)
        if conf > 0.5:  # confidence threshold
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{results.names[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # check if the video writer is None
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()