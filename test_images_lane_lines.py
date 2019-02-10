import glob
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import config

from src import perspective, thresholding, lane_polyfit

if __name__ == "__main__":
    dist_pickle = pickle.load(open(config.get("calibration_filepath"), "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    images = glob.glob('test_images/*.jpg')

    for idx, fname in enumerate(images):
        img = plt.imread(fname)
        plt.imsave(os.path.join("output_images", "original_" + os.path.basename(fname)), img, format="jpg")

        undist = cv2.undistort(img, mtx, dist, None, mtx)
        thresh = thresholding.pipeline(undist, config.get("s_thresh"), config.get("sobel_thresh"), config.get("h_thresh"),
                                       config.get("sobel_kernel"), filename=fname)
        plt.imsave(os.path.join("output_images", "thresh_" + os.path.basename(fname)), thresh, cmap="gray",
                   format="jpg")

        persp, m, minv = perspective.transform(thresh)
        plt.imsave(os.path.join("output_images", "perspective_" + os.path.basename(fname)), persp, cmap="gray",
                   format="jpg")

        window_fit = lane_polyfit.fit_polynomial(persp, undist, minv)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(window_fit, 'Radius of the Curvature = ' + str(int(lane_polyfit.Lane.average_radius)) + '(m)',
                    (50, 100), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(window_fit,
                    "Vehicle is {0}m {1} of center".format(str(round(abs(lane_polyfit.Lane.car_position), 2)),
                                                           "right" if lane_polyfit.Lane.car_position > 0 else "left"),
                    (50, 150), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        plt.imsave(os.path.join("output_images", os.path.basename(fname)), window_fit, format="jpg")

