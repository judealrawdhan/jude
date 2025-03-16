# Traffic Light Control with Truck Detection
# Modified to match the structured method of Raspberry Pi IMX500 examples

import sys
import time
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
from functools import lru_cache
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# GPIO Pins for Traffic Light Module
RED_PIN = 17  # R - Red
GREEN_PIN = 22  # G - Green
YELLOW_PIN = 27  # Y - Yellow
GND = "GND"  # Ground (No control needed)

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(YELLOW_PIN, GPIO.OUT)

def traffic_light_sequence():
    print("🚦 Red Light ON for 30 sec")
    GPIO.output(RED_PIN, GPIO.HIGH)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(YELLOW_PIN, GPIO.LOW)
    time.sleep(30)

    print("🚦 Green Light ON for 15 sec")
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.HIGH)
    GPIO.output(YELLOW_PIN, GPIO.LOW)
    time.sleep(15)

    print("🚦 Yellow Light ON for 3 sec")
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(YELLOW_PIN, GPIO.HIGH)
    time.sleep(3)

    # Reset to Red after cycle
    GPIO.output(RED_PIN, GPIO.HIGH)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    GPIO.output(YELLOW_PIN, GPIO.LOW)

# Classification Variables
last_detections = []
CONFIDENCE_THRESHOLD = 0.20  # Lowering threshold for better detection

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    global last_detections
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        print("❌ Model did not return any outputs. Check if it's running correctly.")
        return last_detections
    last_detections = []
    labels = get_labels()
    print("🔍 Detected objects:")
    for detection in last_detections:
        label = labels[int(detection.category)]
        print(f"✅ Label: {label} | Confidence: {detection.conf:.3f}")
    return last_detections

@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def check_for_truck():
    print("🔍 Checking detected objects...")
    labels = get_labels()
    for detection in last_detections:
        label = labels[int(detection.category)]
        confidence = detection.conf
        print(f"✅ Detected: {label} | Confidence: {confidence:.3f}")
        if "truck" in label.lower() and confidence >= CONFIDENCE_THRESHOLD:
            print("🚛 Truck detected! Changing traffic lights...")
            os.system("aplay truck_alert.wav")
            traffic_light_sequence()
            return True
    return False

if __name__ == "__main__":
    # Model and labels setup
    MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    LABELS_PATH = "/home/jude/projects/RPI_AI_Cam/picamera2/examples/hailo/coco.txt"

    imx500 = IMX500(MODEL_PATH)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "object detection"
    with open(LABELS_PATH, "r") as f:
        intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    picam2.start(config, show_preview=True)
    print("🚦 Waiting for truck detection...")
    
    try:
        while True:
            time.sleep(0.5)
            if check_for_truck():
                time.sleep(5)
    except KeyboardInterrupt:
        print("🚦 Stopping traffic light system")
        GPIO.cleanup()
