import numpy as np
import cv2
import glob
import traceback
import pickle
import matplotlib.pyplot as plt

import config


objp = np.zeros((config.get("x_points") * config.get("y_points"), 3), np.float32)
objp[:, :2] = np.mgrid[0:config.get("x_points"), 0:config.get("y_points")].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/*.jpg')

img_size = []

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = plt.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (config.get("x_points"), config.get("y_points")), None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

try:
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    dist_pickle = {"mtx": mtx, "dist": dist, "ret": ret, "rvecs": rvecs, "tvecs": tvecs}
    pickle.dump(dist_pickle, open(config.get("calibration_filepath"), "wb"))
except:
    traceback.print_exc()
    exit(1)

# Test calibration
images = glob.glob('test_images/*.jpg')

for idx, fname in enumerate(images):
    img = plt.imread(fname)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.waitforbuttonpress()

print("Calibration values saved at " + config.get("calibration_filepath"))
