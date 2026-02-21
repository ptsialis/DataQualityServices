import os
#import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Fixed unlabelled and uncropped images
UNCROPPED_PATH = 'datasets/uncropped_images'
UNLABELLED_PATH = 'datasets/unlabelled_images'

def detect_and_crop(image_path):

    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        print(f"[WARRNING]: Skipping non-image file: {image_path}")
        return None
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[ERROR]: Error opening {image_path}: {e}")
        return None
    
    image = Image.open(image_path).convert('RGB')
    results = model(image)
    
    if results and results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0]) # Detect object
        cropped_img = image.crop((x1, y1, x2, y2)).resize((84,84))

        return cropped_img
    else:
        print(f'[WARNING]: no object detected in {image_path}')
        return None
    
