import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics import RTDETR
import torch
import cv2
import matplotlib.patches as mpatches

rootFolder: str = os.path.join("mixed", "results")

#completeName = "yolo11s_300_100_16"
#completeName = "yolo11n_300_100_16_AdamW_0.0001_0.0001_0"
completeName = "yolo11n_300_100_16"
train_path = os.path.join(rootFolder, completeName, "train")

if not os.path.exists(train_path):
    raise ValueError(f"File path {train_path} isn't present!")

# Load the model
best_model_path = os.path.join(train_path, "weights", "best.pt")
model = YOLO(best_model_path)  # Load the PyTorch model
base_output_dir: str = os.path.join("assets", "output")

# Video input and output paths
# Video 1
#video_name = "walking_porto_downscaled"
#input_video_path = os.path.join("assets", "video_test_1", video_name + ".mp4")
#output_video_path = os.path.join(base_output_dir, video_name + "_" + completeName + ".mp4")

# Video 2
video_name = "walking_porto_1_downscaled"
input_video_path = os.path.join("assets", "video_test_2", video_name + ".mp4")
output_video_path = os.path.join(base_output_dir, video_name + "_" + completeName + ".mp4")

# Video 3
# video_name = "walking_porto_2_downscaled"
# input_video_path = os.path.join("assets", "video_test_2", video_name + ".mp4")
# output_video_path = os.path.join(base_output_dir, video_name + "_" + completeName + ".mp4")

if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# Open the input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Get the video properties (frame width, height, and frames per second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec if necessary
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the model on the current frame
    results = model(frame)  # Perform inference on the frame

    # Parse the results and draw bounding boxes on the frame
    for result in results[0].boxes:  # Access the boxes directly
        x1, y1, x2, y2 = result.xyxy[0]  # Get the bounding box coordinates (x1, y1, x2, y2)
        conf = result.conf[0]  # Confidence score
        cls = result.cls[0]  # Class index

        label = f"{model.names[int(cls)]}: {conf:.2f}"

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Output saved to {output_video_path}")