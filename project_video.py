import glob
import os
import pickle
import numpy as np
import cv2
import config
from moviepy.editor import VideoFileClip

from src import perspective, thresholding, lane_polyfit

dist_pickle = pickle.load(open(config.get("calibration_filepath"), "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


def lane_line_finder_pipeline(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    thresh = thresholding.pipeline(undist, config.get("sobel_thresh"), config.get("schannel_thresh"),
                                   config.get("sobel_kernel"))
    persp, m, minv = perspective.transform(thresh)
    window_fit = lane_polyfit.fit_polynomial(persp, undist, minv)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(window_fit, 'Radius of the Curvature = ' + str(int(lane_polyfit.Lane.average_radius)) + '(m)', (50, 100), font,
                2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(window_fit, "Vehicle is {0}m {1} of center".format(str(round(abs(lane_polyfit.Lane.car_position), 2)),
                                                            "right" if lane_polyfit.Lane.car_position > 0 else "left"),
                (50, 150), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    return window_fit


if __name__ == "__main__":
    for video_path in glob.glob("*.mp4"):
        output = "output_videos/output_" + os.path.splitext(os.path.basename(video_path))[0] + ".mp4"
        clip = VideoFileClip(video_path)
        yellow_clip = clip.fl_image(lane_line_finder_pipeline)
        yellow_clip.write_videofile(output, audio=False)



