from typing import List

from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Load model
model: YOLO = YOLO('yolov8n-seg.pt')
input_file_path = r"C:\Users\Z004SK9V\work\vehicle-detection\Parking-Lot\Images\OVERCAST\2015-11-16\camera1\2015-11-16_0710.jpg"

# predict input file based on model
results: List[Results] = model(input_file_path)

print(results)

# plot the result on input based based on config defined below
image_array = results[0].plot(
    # conf=False,
    # boxes=False
)

# create image from image array and save
image = Image.fromarray(image_array)
image.save('camera1-output.jpg')


# cmd to predict on sample inpur video
# yolo segment predict model=yolov8n-seg.pt source='C:\Users\Z004SK9V\work\vehicle-detection\sample-data\4K_Video_of_Highway_Traffic.mp4' imgsz=320 show=true
