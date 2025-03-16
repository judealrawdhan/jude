import argparse
import sys
import time
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
from typing import List
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from picamera2.devices.imx500.postprocess import softmax

# GPIO Pins for Traffic Light
RED_PIN = 17
YELLOW_PIN = 27
GREEN_PIN = 22

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(YELLOW_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)

def traffic_light_sequence():
    print("🚦 Red Light ON for 30 sec")
    GPIO.output(RED_PIN, GPIO.HIGH)
    GPIO.output(YELLOW_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    time.sleep(30)

    print("🚦 Yellow Light ON for 3 sec")
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.output(YELLOW_PIN, GPIO.HIGH)
    GPIO.output(GREEN_PIN, GPIO.LOW)
    time.sleep(3)

    print("🚦 Green Light ON for 15 sec")
    GPIO.output(RED_PIN, GPIO.LOW)
    GPIO.output(YELLOW_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.HIGH)
    time.sleep(15)

    GPIO.output(RED_PIN, GPIO.HIGH)
    GPIO.output(YELLOW_PIN, GPIO.LOW)
    GPIO.output(GREEN_PIN, GPIO.LOW)

# Classification Variables
last_detections = []
LABELS = None

class Classification:
    def __init__(self, idx: int, score: float):
        self.idx = idx
        self.score = score

def get_label(request: CompletedRequest, idx: int) -> str:
    global LABELS
    if LABELS is None:
        LABELS = intrinsics.labels
        assert len(LABELS) in [1000, 1001], "Labels file should contain 1000 or 1001 labels."
        output_tensor_size = imx500.get_output_shapes(request.get_metadata())[0][0]
        if output_tensor_size == 1000:
            LABELS = LABELS[1:]
    return LABELS[idx]

def parse_classification_results(request: CompletedRequest) -> List[Classification]:
    global last_detections
    np_outputs = imx500.get_outputs(request.get_metadata())
    if np_outputs is None:
        return last_detections
    np_output = np_outputs[0]
    if intrinsics.softmax:
        np_output = softmax(np_output)
    top_index = np.argmax(np_output)
    last_detections = [Classification(top_index, np_output[top_index])]
    return last_detections

def check_for_ambulance():
    for detection in last_detections:
        label = get_label(None, detection.idx)
        print(f"🔍 Detected label: {label} with confidence: {detection.score}")
        if "ambulance" in label.lower():
            print("🚑 Ambulance detected! Changing traffic lights...")
            os.system("aplay ambulance_alert.wav")  # Play alert sound
            traffic_light_sequence()
            return True
    return False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk")
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "classification"
    
    if args.labels:
        with open(args.labels, "r") as f:
            intrinsics.labels = f.read().splitlines()
    else:
        with open("assets/imagenet_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    
    intrinsics.update_with_defaults()
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    
    picam2.start(config, show_preview=True)
    print("🚦 Waiting for ambulance detection...")
    
    try:
        while True:
            time.sleep(0.5)
            if check_for_ambulance():
                time.sleep(5)
    except KeyboardInterrupt:
        print("🚦 Stopping traffic light system")
        GPIO.cleanup()
