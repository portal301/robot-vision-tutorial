# Description: This script captures images from multiple cameras simultaneously using OpenCV and threading.
# Each camera must be connected to a separate USB port on the computer.
# DO NOT USE 'USB hubs' to connect multiple cameras.

import argparse
from multiCameraCapture import MultiCameraCapture

# Example usage
# python step3_simultaneous_capture.py -i [0,1,3]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture images from multiple cameras")
    parser.add_argument("--camera_indice","-i", type=str, default=None, help="folder to save images")
    args = parser.parse_args()

    if args.camera_indice:
        camera_indices = args.camera_indice
        camera_indices = camera_indices[1:-1].split(",")
        camera_indices = [int(i) for i in camera_indices]
    else:
        import sys
        print("Please provide a camera index")
        sys.exit(0)

    multi_camera_capture = MultiCameraCapture()
 
    # Load cameras
    multi_camera_capture.load_cameras(camera_indices, save_folder="./image")
    # Capture images from all cameras
    multi_camera_capture.capture_from_all()
    # Release all cameras
    multi_camera_capture.release_cameras()
