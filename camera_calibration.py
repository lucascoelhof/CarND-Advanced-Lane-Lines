import numpy as np
import cv2
import glob
import traceback
import config


class CameraCalibrationParams:

    def __init__(self):
        _camera_params = config.get("camera_calibration")

        self.x_points = _camera_params["x_points"]
        self.y_points = _camera_params["y_points"]
        self.ret = _camera_params["ret"]
        self.mtx = _camera_params["mtx"]
        self.dist = _camera_params["dist"]
        self.rvecs = _camera_params["rvecs"]
        self.tvecs = _camera_params["tvecs"]

    def update_params(self):
        config.set("camera_calibration", self.__dict__)


cc_params = CameraCalibrationParams()

objp = np.zeros((cc_params.x_points * cc_params.y_points, 3), np.float32)
objp[:, :2] = np.mgrid[0:cc_params.x_points, 0:cc_params.y_points].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/*.jpg')

img_size = []

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img_size = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

try:
# Do camera calibration given object points and image points
    cc_params.ret, cc_params.mtx, cc_params.dist, cc_params.rvecs, cc_params.tvecs =\
        cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
except:
    traceback.print_exc()

cc_params.update_params()
