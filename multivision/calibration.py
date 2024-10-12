import cv2
import numpy as np
import os
import glob
import json
import argparse

class CameraCalibrator:
	def __init__(self, checkerboard_size=(8, 6), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
		self.checkerboard_size = checkerboard_size
		self.criteria = criteria
		self.threedpoints = []  # 3D points in real-world space
		self.twodpoints = []  # 2D points in image plane
		self.objectp3d = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
		self.objectp3d[0, :, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
		self.camera_matrix = None
		self.distortion = None
		self.r_vecs = None
		self.t_vecs = None

	def calibrate(self, images_path):
		# Extract images from the provided path
		images = glob.glob(os.path.join(images_path, '*.jpg'))
		for filename in images:
			image = cv2.imread(filename)
			gray_color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			# Find chessboard corners
			ret, corners = cv2.findChessboardCorners(
				gray_color, self.checkerboard_size,
				cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
			)

			if ret:
				self.threedpoints.append(self.objectp3d)

				# Refine pixel coordinates for 2D points
				corners2 = cv2.cornerSubPix(gray_color, corners, (11, 11), (-1, -1), self.criteria)
				self.twodpoints.append(corners2)

				# Draw and display the corners (for visualization only)
				image = cv2.drawChessboardCorners(image, self.checkerboard_size, corners2, ret)
				cv2.imshow('img', image)
				cv2.waitKey(0)

		cv2.destroyAllWindows()
		h, w = image.shape[:2] 
		# Perform camera calibration by 
		# passing the value of above found out 3D points (threedpoints) 
		# and its corresponding pixel coordinates of the 
		# detected corners (twodpoints) 
		ret, self.camera_matrix, self.distortion, self.r_vecs, self.t_vecs = cv2.calibrateCamera( 
			self.threedpoints, self.twodpoints, gray_color.shape[::-1], None, None) 
		# Displaying results (optional)
		print("Camera matrix:")
		print(self.camera_matrix)
		print("\nDistortion coefficient:")
		print(self.distortion)
		print("\nRotation Vectors:")
		print(self.r_vecs)
		print("\nTranslation Vectors:")
		print(self.t_vecs)

	def save_intrinsics(self, save_path):
		# Save the camera matrix and distortion coefficients to a JSON file
		json_data = {
			"camera_matrix": self.camera_matrix.tolist(),
			"dist_coeff": self.distortion.tolist()
		}
		with open(os.path.join(save_path, "camera_intrinsic.json"), "w") as f:
			json.dump(json_data, f, indent=4)
		print(f"Camera intrinsic parameters saved to {os.path.join(save_path, 'camera_intrinsic.json')}")

# Usage example
if __name__ == "__main__":
    calibrator = CameraCalibrator()

    # Step 1: Find corners in calibration images
    calibrator.calibrate(os.path.join("calibration", "cam0"))

    # Step 2: Calibrate the camera (assuming all images are of the same resolution)
    # sample_image = cv2.imread(os.path.join("calibration", "cam0", "sample.jpg"))
    # calibrator.calibrate_camera(sample_image.shape[:2])

    # Step 3: Save intrinsic parameters
    calibrator.save_intrinsics(save_path=os.path.join("calibration", "cam0"))
