Project Overview
This project demonstrates the implementation of advanced object detection techniques using the YOLOv8 model in Python. It encompasses three key applications:

Object Detection in Images
Object Detection in Video Files
Real-Time Object Detection Using Webcam
The project is structured into three distinct Python scripts, each tailored to a specific medium (images, videos, and real-time webcam feed), showcasing the versatility and efficiency of YOLOv8 in various contexts.

Features
Object Detection in Images (YOLOv8_picture_OD.py): This script processes static images to detect and label objects using the YOLOv8 model. It is ideal for analyzing photographs or any single-frame content.

Object Detection in Videos (YOLO_Video_OD.py): Designed to handle video files, this script processes each frame of a video, detecting and annotating objects throughout the video duration. It demonstrates the model's capability in handling dynamic, multi-frame content.

Real-Time Object Detection (YOLO_OD_Live.py): This script utilizes a webcam to capture live video feed and perform real-time object detection. It highlights the model's application in scenarios requiring immediate analysis and response, such as surveillance or interactive systems.

Technologies
YOLOv8: The latest iteration of the YOLO (You Only Look Once) object detection models, known for its speed and accuracy in real-time object detection.
Python: The scripts are written in Python, a widely-used programming language in data science and machine learning.
OpenCV (cv2): Used for handling image and video operations, including reading, processing, and displaying content.
Usage
Each script can be run independently, depending on the user’s requirement:

For image and video object detection, provide the path to the input file.
For real-time detection, ensure a webcam is connected and operational.
Applications
This project can be utilized in various domains, including surveillance, autonomous vehicle systems, traffic monitoring, and interactive media. It serves as a practical example for those interested in computer vision, machine learning, and real-world applications of deep learning technologies.

Getting Started
To get started with this project:

Clone the repository.
Install the required dependencies: ultralytics, opencv-python, and numpy.
Run the desired script with the appropriate arguments (for image and video detection).
