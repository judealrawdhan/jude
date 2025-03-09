#!/usr/bin/env python3
import threading
import argparse
import sys
import time
import os
import cv2
import numpy as np
from typing import List
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics
from picamera2.devices.imx500.postprocess import softmax

# === CONFIGURATION ===
AMBULANCE_THRESHOLD = 0.7
detection_lock = threading.Lock()
ambulance_detected = False
last_detections = []
LABELS = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, 
                      default="/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk",
                      help="Path to model file")
    parser.add_argument("--fps", type=int, help="Frame rate")
    parser.add_argument("-s", "--softmax", action=argparse.BooleanOptionalAction, 
                      help="Enable softmax post-processing")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                      help="Preserve aspect ratio")
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
            LABELS = LABELS[1:]
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

    for detection in last_detections:
        label = get_label(request, detection.idx).lower()
        if "ambulance" in label and detection.score >= AMBULANCE_THRESHOLD:
            with detection_lock:
                ambulance_detected = True
            break
    return last_detections

def draw_classification_results(request: CompletedRequest, results: List[Classification], stream: str = "main"):
    with MappedArray(request, stream) as m:
        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))
            text_left, text_top = b_x, b_y + 20
        else:
            text_left, text_top = 0, 0
        
        for index, result in enumerate(results):
            label = get_label(request, result.idx)
            text = f"{label}: {result.score:.3f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = text_left + 5
            text_y = text_top + 15 + index * 20
            overlay = m.array.copy()
            cv2.rectangle(overlay, (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255), cv2.FILLED)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)
            cv2.putText(m.array, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def parse_and_draw_classification_results(request: CompletedRequest):
    results = parse_classification_results(request)
    draw_classification_results(request, results)

def check_ambulance():
    global ambulance_detected
    if os.path.exists("/tmp/ambulance.trigger"):
        return True
    with detection_lock:
        current = ambulance_detected
        ambulance_detected = False
        return current

if __name__ == "__main__":
    args = get_args()
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "classification"
    elif intrinsics.task != "classification":
        print("Network is not a classification task", file=sys.stderr)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    
    picam2.pre_callback = parse_and_draw_classification_results
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        picam2.stop()
        print("Camera stopped")
