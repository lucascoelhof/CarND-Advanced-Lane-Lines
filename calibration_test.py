import os
import pickle

import cv2
import matplotlib.pyplot as plt
import config


if __name__ == "__main__":
    dist_pickle = pickle.load(open(config.get("calibration_filepath"), "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    fname = "camera_cal/calibration1.jpg"
    img = plt.imread(fname)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    plt.imsave(os.path.join("output_images", "undist_" + os.path.basename(fname)), undist, format="jpg")
