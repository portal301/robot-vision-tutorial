# theory: https://xoft.tistory.com/80

import cv2
import numpy as np
import os
import json

class MonoCamera:
    def __init__(self, camera_name):
        self.camera_name = camera_name
        self.camera_index, self.camera_matrix, self.dist_coeffs, self.coordinate = self.load_camera_calibration(camera_name)
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def load_camera_calibration(self,camera_name):
        base_path = os.path.join("calibration",camera_name)
        with open(os.path.join(base_path,"camera_intrinsic.json")) as f:
            data = json.load(f)
            camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
            dist_coeffs = np.array(data["dist_coeff"], dtype=np.float32)

        with open(os.path.join("calibration","camera_index_map.json")) as f:
            camera_index_map = json.load(f)
            camera_index = camera_index_map[camera_name]

        with open(os.path.join(base_path,"coordinate.json")) as f:
            coordinate=json.load(f)

        return camera_index, camera_matrix, dist_coeffs, coordinate

ocam0 = MonoCamera(camera_name="ocam0")
ocam1 = MonoCamera(camera_name="ocam1")
ocam2 = MonoCamera(camera_name="ocam2")

stereo_camera_left = ocam2
stereo_camera_right = ocam0


# Load the stereo calibration parameters (you need to pre-calibrate your stereo cameras)
# Assuming these are loaded from your calibration process

R = np.array([0,0,0], dtype=np.float64) # Rotation matrix from stereo calibration
# T = np.array(ocam2.coordinate["tvec"]) - np.array(ocam0.coordinate["tvec"])
# T = np.array([0.37,-0.045,0])
T = np.array([0.05,0,0], dtype=np.float64) # Translation vector from stereo calibration

# Open video capture for two cameras
cam_left = cv2.VideoCapture(stereo_camera_left.camera_index)  # Change the index depending on your cameras
cam_right = cv2.VideoCapture(stereo_camera_right.camera_index)

#5MP: 2592x1944(ocam)
#2MP: 1920x1080
# Set the same resolution for both cameras
# width, height = 640, 480
width, height = 2592, 1944
# width, height = 1920, 1080

cam_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# Stereo rectification (rectify the images using the calibration parameters)
# Use stereoRectify to compute the rectification and projection matrices
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    stereo_camera_left.camera_matrix, stereo_camera_left.dist_coeffs,
    stereo_camera_right.camera_matrix, stereo_camera_right.dist_coeffs,
    (width, height), R, T, alpha=1)


# Precompute the rectification map to apply in real-time for left and right images
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    stereo_camera_left.camera_matrix, stereo_camera_left.dist_coeffs, R1, P1, (width, height), cv2.CV_32FC1)

right_map1, right_map2 = cv2.initUndistortRectifyMap(
    stereo_camera_right.camera_matrix, stereo_camera_right.dist_coeffs, R2, P2, (width, height), cv2.CV_32FC1)

# StereoBM object for disparity computation
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
stereo = cv2.StereoBM_create(numDisparities=16*4, blockSize=31)


# Create StereoSGBM object
stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*4,  # Number of disparities to search (multiple of 16)
    blockSize=31,        # Block size (odd number, typically between 5-21)
    P1=8 * 3 * 5 ** 2,  # Penalty for disparity change by +/- 1
    P2=32 * 3 * 5 ** 2, # Penalty for disparity change by more than 1
    disp12MaxDiff=1,
    uniquenessRatio=1,
    speckleWindowSize=100,
    speckleRange=32
)

while True:
    ret_left, frame_left = cam_left.read() # ret_left: success/fail, frame_left: frame
    ret_right, frame_right = cam_right.read()

    if ret_left and ret_right:
        # Rectify the images using the precomputed maps
        rectified_left = cv2.remap(frame_left, left_map1, left_map2, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(frame_right, right_map1, right_map2, cv2.INTER_LINEAR)

        # Convert to grayscale
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        # disparity = stereo.compute(gray_left, gray_right)
        # Inside the loop, use StereoSGBM to compute the disparity map
        disparity = stereo_sgbm.compute(gray_left, gray_right)

        # print min and max disparity
        print(f"Min disparity: {disparity.min()}, Max disparity: {disparity.max()}")

        # # Covert disparity to depth
        # depth_map = cv2.reprojectImageTo3D(disparity, Q)

        # # Normalized depth for display (optional)
        # depth_view = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Normalize the disparity for display
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Show the images and disparity map
        cv2.imshow("Left Camera", cv2.resize(rectified_left, (640, 480)))
        cv2.imshow("Right Camera", cv2.resize(rectified_right, (640, 480)))
        cv2.imshow("Disparity Map", cv2.resize(disp_norm, (640, 480)))
        # cv2.imshow("Depth Map", cv2.resize(depth_view, (640, 480)))

        # Break the loop on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error capturing video stream")

# Release video capture objects
cam_left.release()
cam_right.release()
cv2.destroyAllWindows()
