import glob
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

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    for idx, fname in enumerate(images):
        img = plt.imread(fname)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        thresh = thresholding.pipeline(dst, config.get("sobel_thresh"), config.get("schannel_thresh"))
        persp = perspective.transform(thresh)
        window_fit = lane_polyfit.fit_polynomial(persp)
        print("left: {0}, right: {1}".format(lane_polyfit.Lane.left.radius_of_curvature,
                                             lane_polyfit.Lane.right.radius_of_curvature))
        print("Vehicle is {0} m {1} from center".format(str(round(abs(lane_polyfit.Lane.car_position), 2)),
                                                        "right" if lane_polyfit.Lane.car_position > 0 else "left"))

        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(window_fit, cmap='gray')
        ax2.set_title('Perspective Image', fontsize=30)
        plt.draw()
        plt.waitforbuttonpress()
        ax1.cla()
        ax2.cla()

