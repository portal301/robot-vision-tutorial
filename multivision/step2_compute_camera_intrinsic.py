# Description: This script captures images from multiple cameras simultaneously using OpenCV and threading.
# Each camera must be connected to a separate USB port on the computer.
# DO NOT USE 'USB hubs' to connect multiple cameras.

import os
import argparse
from calibration import CameraCalibrator

# Example usage
# python step2_compute_camera_intrinsic.py -c "ocam2"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture images from multiple cameras")
    parser.add_argument("--camera_name","-c", type=str, default=None, help="folder to the saved images")

    args = parser.parse_args()
    if args.camera_name:
        save_folder = os.path.join(".","calibration",args.camera_name)
    else:
        import sys
        print("Please provide a camera name")
        sys.exit(0)

    calibrator = CameraCalibrator()

    # Step 2: Calibrate the camera (assuming all images are of the same resolution)
    calibrator.calibrate(save_folder)

    # Step 3: Save intrinsic parameters
    calibrator.save_intrinsics(save_path=save_folder)
