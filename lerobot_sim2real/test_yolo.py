#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot with independent YOLO streaming display
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
Keyboard control is identical to 5_so100_keyboard_ee_control.py

YOLO stream displays object detection but does NOT control the robot
Video stream and robot control are completely independent
"""

import time
import logging
import traceback
import math
import cv2
import numpy as np
import threading
from ultralytics import YOLOE

def video_stream_loop(model, cap, target_objects=None):
    """
    Independent video streaming loop that only displays object detection
    Does not control the robot - purely for visual feedback
    """
    print("Starting YOLO video stream...")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame not available")
                continue

            results = model(frame)
            if not results or not hasattr(results[0], 'boxes') or not results[0].boxes:
                # No objects detected - show original frame
                annotated_frame = frame
            else:
                # Show detection results
                annotated_frame = results[0].plot()
            
            # Show detection results in a window
            cv2.imshow("YOLO Live Detection", annotated_frame)
            
            # Allow quitting vision mode with 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                break
                
        except Exception as e:
            print(f"Video stream error: {e}")
            break
    
    print("Video stream ended")
    cv2.destroyAllWindows()

def main():
    """Main function"""
    print("LeRobot Keyboard Control + Independent YOLO Display")
    print("="*60)
    
    try:
        
        # Initialize YOLO and camera
        model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes
        
        # Get detection targets from user input
        print("\n" + "="*60)
        print("YOLO Detection Target Setup")
        print("="*60)
        target_input = input("Enter objects to detect (separate multiple objects with commas, e.g., bottle,cup,mouse): ").strip()
        
        # If Enter is pressed directly, use default targets
        if not target_input:
            target_objects = ["black robot manipulator", "red square cube"]
            print(f"Using default targets: {target_objects}")
        else:
            # Parse multiple objects separated by commas
            target_objects = [obj.strip() for obj in target_input.split(',') if obj.strip()]
            print(f"Detection targets: {target_objects}")
        
        # Set text prompt to detect the specified objects
        model.set_classes(target_objects, model.get_text_pe(target_objects))
        
        # List available cameras and prompt user
        def list_cameras(max_index=8):
            available = []
            for idx in range(max_index):
                cap_test = cv2.VideoCapture(idx)
                if cap_test.isOpened():
                    available.append(idx)
                    cap_test.release()
            return available
        cameras = list_cameras()
        if not cameras:
            print("No cameras found!")
            return
        print(f"Available cameras: {cameras}")
        selected = int(input(f"Select camera index from {cameras}: "))
        cap = cv2.VideoCapture(selected)
        if not cap.isOpened():
            print("Camera not found!")
            return
        print("Video stream:")
        print("- Independent YOLO detection display (no robot control)")
        print("- Q (in YOLO window): Exit video stream")
        print("="*60)
        print("Note: Video stream and keyboard control are completely independent")
        
        # Start video stream in a separate thread
        video_stream_loop(model, cap, target_objects)
        
        cap.release()
        cv2.destroyAllWindows()
        print("Program ended")
        
    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Is the robot correctly connected")
        print("2. Is the USB port correct")
        print("3. Do you have sufficient permissions to access USB device")
        print("4. Is the robot correctly configured")

if __name__ == "__main__":
    main() 