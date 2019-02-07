import config
import numpy as np
import cv2


def transform(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(config.get("trapezoid"))
    dest = np.float32(config.get("dest_points"))
    m = cv2.getPerspectiveTransform(src, dest)
    minv = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_NEAREST)
    return warped, m, minv
