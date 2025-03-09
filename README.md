#!/usr/bin/env python3
import threading
import argparse
import sys
import time
from typing import List
import cv2
import numpy as np
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from picamera2.devices.imx500.postprocess import softmax

# === AMBULANCE DETECTION FLAGS ===
ambulance_detected = False
AMBULANCE_THRESHOLD = 0.7
detection_lock = threading.Lock()

last_detections = []
LABELS = None

# ========== FUNCTION DEFINITIONS ==========
def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, 
                      default="/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk",
                      help="Path of the model")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("-s", "--softmax", action=argparse.BooleanOptionalAction, 
                      help="Add post-process softmax")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                      help="Preprocess with aspect ratio preservation")
    parser.add_argument("--labels", type=str, help="Path to labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                      help="Print network intrinsics")
    return parser.parse_args()

class Classification:
    def __init__(self, idx: int, score: float):
        self.idx = idx
        self.score = score

def get_label(request: CompletedRequest, idx: int) -> str:
    global LABELS
    if LABELS is None:
        LABELS = intrinsics.labels
        assert len(LABELS) in [1000, 1001], "Invalid labels file"
        output_tensor_size = imx500.get_output_shapes(request.get_metadata())[0][0]
        if output_tensor_size == 1000:
            LABELS = LABELS[1:]  # Remove background label
    return LABELS[idx]

def parse_classification_results(request: CompletedRequest) -> List[Classification]:
    global last_detections, ambulance_detected
    np_outputs = imx500.get_outputs(request.get_metadata())
    
    if np_outputs is None:
        return last_detections
        
    np_output = np_outputs[0]
    if intrinsics.softmax:
        np_output = softmax(np_output)
    
    top_indices = np.argpartition(-np_output, 3)[:3]
    top_indices = top_indices[np.argsort(-np_output[top_indices])]
    last_detections = [Classification(index, np_output[index]) for index in top_indices]

    # Ambulance check
    for detection in last_detections:
        label = get_label(request, detection.idx).lower()
        if "ambulance" in label and detection.score >= AMBULANCE_THRESHOLD:
            with detection_lock:
                ambulance_detected = True
            break
    return last_detections

def draw_classification_results(request: CompletedRequest, results: List[Classification], stream: str = "main"):
    """Draw classification results on frame"""
    with MappedArray(request, stream) as m:
        # Add your drawing implementation here
        pass

def parse_and_draw_classification_results(request: CompletedRequest):
    results = parse_classification_results(request)
    draw_classification_results(request, results)

def check_ambulance():
    global ambulance_detected
    with detection_lock:
        current_status = ambulance_detected
        ambulance_detected = False
    return current_status

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    args = get_args()
    
    # Initialize IMX500 and intrinsics
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "classification"
    elif intrinsics.task != "classification":
        print("Network is not a classification task", file=sys.stderr)
        exit()

    # Rest of initialization code
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    picam2.pre_callback = parse_and_draw_classification_results
    
    while True:
        time.sleep(0.1)
