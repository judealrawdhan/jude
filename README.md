import argparse
import sys
import time
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
from functools import lru_cache
from typing import List
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
CONFIDENCE_THRESHOLD = 0.50  # Adjust confidence threshold for trucks

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def parse_detections(metadata: dict):
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_detections
        )[0]
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h
        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections

@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def check_for_truck():
    """Check if a truck is detected and change traffic lights."""
    print("🔍 Checking detected objects...")
    labels = get_labels()
    for detection in last_detections:
        label = labels[int(detection.category)]
        confidence = detection.conf
        print(f"✅ Detected: {label} | Confidence: {confidence:.3f}")
        if "truck" in label.lower() and confidence >= CONFIDENCE_THRESHOLD:
            print("🚛 Truck detected! Changing traffic lights...")
            os.system("aplay truck_alert.wav")  # Play alert sound if needed
            traffic_light_sequence()
            return True
    return False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "object detection"
    if args.labels:
        with open(args.labels, "r") as f:
            intrinsics.labels = f.read().splitlines()
    else:
        with open("/home/jude/projects/RPI_AI_Cam/picamera2/examples/imx500/imx500_object_detection_demo.py", "r") as f:
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
