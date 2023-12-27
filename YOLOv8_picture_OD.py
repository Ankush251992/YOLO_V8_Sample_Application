
from ultralytics import YOLO
import cv2
import argparse
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load YOLOv8 model
print("[INFO] loading YOLOv8 from disk...")
model = YOLO("yolov8n.pt")

# run predictions
results = model([args["image"]])

# process results
for result in results:
    image = cv2.imread(args["image"])
    for box in result.boxes:
        xyxy = box.xyxy.cpu().numpy().squeeze()
        x1, y1, x2, y2 = map(int, [xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
        conf = float(box.conf)
        cls = int(box.cls)
        if conf > 0.5:  # confidence threshold
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{result.names[cls]}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()