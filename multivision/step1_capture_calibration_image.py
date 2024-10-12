# Description: This script captures images from multiple cameras simultaneously using OpenCV and threading.
# Each camera must be connected to a separate USB port on the computer.
# DO NOT USE 'USB hubs' to connect multiple cameras.

import os
import argparse
from multiCameraCapture import MultiCameraCapture


# Example usage
# python step1_capture_calibration_image.py -i 0 -n 10 -c cam0 -p 640x480
# python step1_capture_calibration_image.py -i 0 -n 10 -c cam0 -p 2592x1944
# width, height = 2592, 1944
# width, height = 1920, 1080

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture images from multiple cameras")
    parser.add_argument("--index","-i", type=int, default=0, help="camera index to capture from")
    parser.add_argument("--num_images","-n", type=int, default=10, help="number of images to capture")
    parser.add_argument("--camera_name","-c", type=str, default=None, help="folder to save images")
    parser.add_argument("--pixel","-p", type=str, default=None, help="resolution of the image")

    args = parser.parse_args()
    camera_index = args.index
    num_images = args.num_images
    if args.camera_name:
        save_folder = os.path.join(".","calibration",args.camera_name)
    else:
        save_folder = os.path.join(".","calibration",f"cam{camera_index}")

    if args.pixel:
        width_px, height_px = args.pixel.split("x")
        width_px, height_px = int(width_px), int(height_px)
    else:
        width_px, height_px = 640, 480


    multi_camera_capture = MultiCameraCapture()
    multi_camera_capture.capture_calibration_images(camera_index=camera_index, num_images=num_images, save_folder=save_folder, image_size=(width_px, height_px))


