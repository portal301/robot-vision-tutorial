import cv2 as cv
import numpy as np
import sys
import json
import imageio

def get_camera_params(path):
    # open a json file from path and return camera matrix and distortion parameters.
    with open(path) as f:
        data = json.load(f)
        cmtx = data['camera_matrix']
        dist = data['dist_coeff'] 
    return np.array(cmtx), np.array(dist)


def get_qr_coords(cmtx, dist, points):

    #Selected coordinate points for each corner of QR code.
    qr_edges = np.array([[0,0,0],
                         [0,1,0],
                         [1,1,0],
                         [1,0,0]], dtype = 'float32').reshape((4,1,3))

    #determine the orientation of QR code coordinate system with respect to camera coorindate system.
    ret, rvec, tvec = cv.solvePnP(qr_edges, points, cmtx, dist)

    #Define unit xyz axes. These are then projected to camera view using the rotation matrix and translation vector.
    unitv_points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    if ret:
        points, jac = cv.projectPoints(unitv_points, rvec, tvec, cmtx, dist)
        return points, rvec, tvec

    #return empty arrays if rotation and translation values not found
    else: return [], [], []
def run_frame(cmtx, dist, in_source, gif_filename='output.gif'):
    cap = cv.VideoCapture(in_source)
    qr = cv.QRCodeDetector()
    frames = []

    while True:
        ret, img = cap.read()
        if not ret: break

        ret_qr, points = qr.detect(img)

        if ret_qr:
            axis_points, rvec, tvec = get_qr_coords(cmtx, dist, points)

            # BGR color format
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]

            if len(axis_points) > 0:
                axis_points = axis_points.reshape((4, 2))
                origin = (int(axis_points[0][0]), int(axis_points[0][1]))

                for p, c in zip(axis_points[1:], colors[:3]):
                    p = (int(p[0]), int(p[1]))

                    # Skip cases where the projected point overflows integer value
                    if origin[0] > 5 * img.shape[1] or origin[1] > 5 * img.shape[1]: break
                    if p[0] > 5 * img.shape[1] or p[1] > 5 * img.shape[1]: break

                    cv.line(img, origin, p, c, 5)

            # Capture the frame for GIF
            frames.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        cv.imshow('frame', img)
        k = cv.waitKey(20)
        if k == 27: break  # 27 is ESC key.

    cap.release()
    cv.destroyAllWindows()

    # Save the captured frames as a GIF
    imageio.mimsave(gif_filename, frames, fps=30)

if __name__ == '__main__':
    # Read camera intrinsic parameters.
    cmtx, dist = get_camera_params('camera_params/camera_intrinsic.json')

    input_source = 'video/sample.mp4'

    # To use webcam, set input_source to 0.
    # ex) python qr_tracking.py 0
    if len(sys.argv) > 1:
        input_source = int(sys.argv[1])

    run_frame(cmtx, dist, input_source)
