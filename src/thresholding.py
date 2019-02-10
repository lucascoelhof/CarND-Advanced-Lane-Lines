import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def pipeline(img, s_thresh=[30, 100], sx_thresh=[20, 100], h_thresh=[10, 30], sobel_kernel=3, filename=""):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    if filename:
        plt.imsave(os.path.join("output_images", "sobel_" + os.path.basename(filename)), sxbinary, cmap="gray", format="jpg")

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    if filename:
        plt.imsave(os.path.join("output_images", "s_channel_" + os.path.basename(filename)), s_binary, cmap="gray", format="jpg")

    # Stack each channel
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary
