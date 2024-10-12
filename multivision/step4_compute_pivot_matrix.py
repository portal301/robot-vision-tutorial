import cv2
import cv2.aruco as aruco
import numpy as np
import os
import json


# ------------------------------
# ENTER YOUR PARAMETERS HERE:
# ARUCO_DICT = cv2.aruco.DICT_6X6_250
ARUCO_DICT = cv2.aruco.DICT_7X7_1000
SQUARES_HORIZONTALLY = 7
SQUARES_VERTICALLY = 5
SQUARE_LENGTH = 0.035 # Square length in meters (35 mm)
MARKER_LENGTH = 0.0175 # Marker length in meters (17.5 mm)
# LENGTH_PX = 9933   # total length of the page in pixels
LENGTH_PX = 3508   # total vertical length of the page in pixels
MARGIN_PX = 200    # size of the margin in pixels
SAVE_NAME = 'ChArUco_A4_Board.png'
# ------------------------------

def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    # cv2.imshow("img", img)
    # cv2.waitKey(1000)
    cv2.imwrite(os.path.join("board","ChArUco_A4_Board.png"), img)
    return board

# Detect the ChArUco board and estimate its 3D pose
def detect_and_estimate_charuco_pose(image, charuco_board, camera_matrix, dist_coeffs):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the image
    # corners, ids, rejected_img_points = aruco.detectMarkers(gray, charuco_board.dictionary)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    # Detect ArUco markers in the image
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        # Refine the marker detection by removing false positives
        aruco.refineDetectedMarkers(gray, charuco_board, corners, ids, rejected_img_points)

        # Interpolate ChArUco corners based on detected markers
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=charuco_board)

        # If enough ChArUco corners are detected, estimate the board pose
        if retval > 0:
            # Estimate pose of ChArUco board (rvec = rotation vector, tvec = translation vector)
            # Initialize the rotation and translation vectors
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))

            # Estimate pose of ChArUco board
            success = aruco.estimatePoseCharucoBoard(
                charucoCorners=charuco_corners, charucoIds=charuco_ids,
                board=charuco_board, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs,
                rvec=rvec, tvec=tvec)
            
            if success:
                # Draw the 3D axis on the image
                cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.1) 
                return rvec, tvec
            else:
                print("Pose estimation failed.")
                return None, None
        else:
            print("Not enough ChArUco corners detected.")
            return None, None
    else:
        print("No ArUco markers detected.")
        return None, None


def compute_charco_pose_from_camera(camera_name):
    base_path = os.path.join("calibration",camera_name)
    with open(os.path.join(base_path,"camera_intrinsic.json")) as f:
        data = json.load(f)
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(data["dist_coeff"], dtype=np.float32)

    with open(os.path.join("calibration","camera_index_map.json")) as f:
        camera_index_map = json.load(f)
        camera_index = camera_index_map[camera_name]
    # Read the image containing the ChArUco board
    image = cv2.imread(os.path.join("image",f"cam_{camera_index}_image.jpg"))

    # Detect and estimate the 3D pose of the ChArUco board
    rvec, tvec = detect_and_estimate_charuco_pose(image, charuco_board, camera_matrix, dist_coeffs)

    # Display the result
    if rvec is not None and tvec is not None:
        cv2.imshow('Detected ChArUco Board', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # convert rvec and tvec to 4x4 matrix
    rvec = np.array(rvec)
    tvec = np.array(tvec)
    rmat, _ = cv2.Rodrigues(rvec)
    T = np.hstack((rmat, tvec))
    T = np.vstack((T, np.array([0, 0, 0, 1])))

    output = {
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "pivot_matrix": T.tolist()
    }

    with open(os.path.join(base_path,"coordinate.json"), "w") as f:
        json.dump(output, f, indent=4)

    return output

# Main function
if __name__ == "__main__":
    # Create the ChArUco board
    charuco_board = create_and_save_new_board()

    cameras = ["ocam0", "ocam1", "ocam2"]
    coordinates = {}
    for camera_name in cameras:
        coordinates[camera_name] = compute_charco_pose_from_camera(camera_name)
        print(np.array(coordinates[camera_name]["pivot_matrix"]))




