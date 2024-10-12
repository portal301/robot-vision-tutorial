# Description: This script captures images from multiple cameras simultaneously using OpenCV and threading.
# Each camera must be connected to a separate USB port on the computer.
# DO NOT USE 'USB hubs' to connect multiple cameras.
import os
from multiCameraCapture import MultiCameraCapture
import json

# Example usage
# python step0_make_camera_index_map.py
if __name__ == "__main__":
    multi_camera_capture = MultiCameraCapture()
    print("Scanning for available cameras...")
    camera_indices = multi_camera_capture.scan_available_cameras()
    print(f"Scan completed. Available cameras: {camera_indices}")
 
    # Load cameras
    multi_camera_capture.load_cameras(camera_indices, save_folder="./image")
    # Capture images from all cameras
    multi_camera_capture.capture_from_all()
    # Release all cameras
    multi_camera_capture.release_cameras()

    camera_index_map = {
        "ocam0": 0,
        "ocam1": 1,
        "integrated_camera": 2,
        "ocam2": 3
    }

    # Save the camera index map.json
    with open(os.path.join("calibration","camera_index_map.json"), "w") as f:
        json.dump(camera_index_map, f, indent=4)

