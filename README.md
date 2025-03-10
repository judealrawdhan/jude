#!/usr/bin/env python3
import threading
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from picamera2.devices import IMX500
import numpy as np
import os

# ====== CONFIGURATION ======
PINS = {'red': 17, 'yellow': 27, 'green': 22}
FIRMWARE_PATH = "/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk"
DETECTION_THRESHOLD = 0.7  # 70% confidence

# ====== SHARED STATE ======
ambulance_detected = False
system_lock = threading.Lock()

# ====== GPIO INITIALIZATION ======
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in PINS.values():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# ====== CAMERA FUNCTIONS ======
def initialize_camera():
    """Initialize IMX500 camera and Picamera2."""
    if not os.path.exists(FIRMWARE_PATH):
        print(f"âŒ Missing firmware: {FIRMWARE_PATH}")
        print("Install with: sudo apt install imx500-firmware")
        return None, None
    
    try:
        print("ðŸ”„ Initializing IMX500 camera...")
        imx500 = IMX500(FIRMWARE_PATH)
        picam2 = Picamera2(imx500.camera_num)
        config = picam2.create_preview_configuration()
        picam2.configure(config)
        print("âœ… Camera successfully initialized!")
        return imx500, picam2
    except Exception as e:
        print(f"âŒ Camera error: {str(e)}")
        return None, None

def detect_ambulance(imx500, picam2):
    """Continuously capture frames and check for ambulance detection."""
    global ambulance_detected
    print("ðŸš€ Starting IMX500 AI detection...")

    try:
        picam2.start()
        print("âœ… Camera started successfully")

        while True:
            print("ðŸ“¸ Capturing frame...")
            request = picam2.capture_request()

            if not request:
                print("âŒ Failed to capture request")
                time.sleep(0.1)
                continue  # Skip and try again

            outputs = imx500.get_outputs(request.get_metadata())

            if outputs:
                print("ðŸ” Model processed the frame")
                output = outputs[0]
                top_indices = np.argpartition(-output, 3)[:3]

                for idx in top_indices:
                    label = imx500.network_intrinsics.labels[idx].lower()
                    confidence = output[idx]
                    print(f"ðŸ›‘ Detected: {label} ({confidence:.2f})")

                    if "ambulance" in label and confidence > DETECTION_THRESHOLD:
                        with system_lock:
                            ambulance_detected = True
                            print("ðŸš‘ AMBULANCE DETECTED! Prioritizing traffic light.")

            else:
                print("âš ï¸ No objects detected in this frame.")

            request.release()
            time.sleep(0.1)
    except Exception as e:
        print(f"âŒ Error in detect_ambulance(): {str(e)}")
    finally:
        picam2.stop()
        print("ðŸ“· Camera stopped")

# ====== TRAFFIC LIGHT FUNCTIONS ======
def traffic_control():
    """Control traffic light behavior based on ambulance detection."""
    global ambulance_detected
    try:
        while True:
            with system_lock:
                current_state = ambulance_detected
                ambulance_detected = False  # Reset after checking
                
            if current_state:
                # Emergency sequence: Green for ambulance
                GPIO.output(PINS['green'], GPIO.HIGH)
                GPIO.output(PINS['yellow'], GPIO.LOW)
                GPIO.output(PINS['red'], GPIO.LOW)
                print("ðŸš¦ EMERGENCY MODE: GREEN LIGHT FOR AMBULANCE")
                time.sleep(15)  # Hold green for 15 seconds
                
                # Transition back to normal
                GPIO.output(PINS['green'], GPIO.LOW)
                GPIO.output(PINS['yellow'], GPIO.HIGH)
                time.sleep(3)
                GPIO.output(PINS['yellow'], GPIO.LOW)
                GPIO.output(PINS['red'], GPIO.HIGH)
                time.sleep(15)  # Red phase before normal cycle
            else:
                # Normal cycle
                GPIO.output(PINS['green'], GPIO.HIGH)
                print("ðŸš¦ NORMAL TRAFFIC: GREEN LIGHT")
                time.sleep(15)
                GPIO.output(PINS['green'], GPIO.LOW)
                
                GPIO.output(PINS['yellow'], GPIO.HIGH)
                print("ðŸš¦ NORMAL TRAFFIC: YELLOW LIGHT")
                time.sleep(3)
                GPIO.output(PINS['yellow'], GPIO.LOW)
                
                GPIO.output(PINS['red'], GPIO.HIGH)
                print("ðŸš¦ NORMAL TRAFFIC: RED LIGHT")
                time.sleep(15)
                GPIO.output(PINS['red'], GPIO.LOW)
                
    except KeyboardInterrupt:
        GPIO.cleanup()

# ====== MAIN EXECUTION ======
if __name__ == "__main__":
    print("ðŸ”„ Starting Integrated System...")
    
    imx500, picam2 = initialize_camera()
    if not imx500:
        print("âŒ Camera initialization failed. Exiting.")
        exit(1)

    # Start threads
    cam_thread = threading.Thread(target=detect_ambulance, args=(imx500, picam2), daemon=True)
    traffic_thread = threading.Thread(target=traffic_control, daemon=True)
    
    cam_thread.start()
    traffic_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nðŸ”´ Shutting down...")
        picam2.stop()
        GPIO.cleanup()

              
    
