import numpy as np
import cv2
import config
import collections


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # historical fits
        self.hist_fit = collections.deque(maxlen=30)
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


class Lane:
    left = Line()
    right = Line()
    car_position = 0
    average_radius = 0


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def update_curvature(img_shape):
    bondary_lenght = config.get("bondary_lenght")
    ym_per_pix = bondary_lenght / img_shape[0]
    left_y = img_shape[0]
    right_y = img_shape[0]
    left_curverad = ((1 + (2 * Lane.left.current_fit[0] * left_y * ym_per_pix + Lane.left.current_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * Lane.left.current_fit[0])
    right_curverad = ((1 + (2 * Lane.right.current_fit[0] * right_y * ym_per_pix + Lane.right.current_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * Lane.right.current_fit[0])
    Lane.left.radius_of_curvature = left_curverad
    Lane.right.radius_of_curvature = right_curverad
    Lane.average_radius = (left_curverad + right_curverad)/2
    return left_curverad, right_curverad


def find_car_position(img_shape):
    lane_width = config.get("lane_width")
    xm_per_pix = lane_width/(Lane.left.recent_xfitted[0] - Lane.right.recent_xfitted[0])
    return ((Lane.left.recent_xfitted[0] + Lane.right.recent_xfitted[0])/2 - img_shape[1]/2) * xm_per_pix


def find_best_fit():
    avg = np.zeros_like(Lane.left.current_fit)
    for fit in Lane.left.hist_fit:
        avg = np.add(fit, avg)
    Lane.left.best_fit = avg / len(Lane.left.hist_fit)

    avg = np.zeros_like(Lane.right.current_fit)
    for fit in Lane.right.hist_fit:
        avg = np.add(fit, avg)
    Lane.right.best_fit = avg / len(Lane.right.hist_fit)


def fit_polynomial(binary_warped, undist, minv):
    # Find our lane pixels first

    img_shape = binary_warped.shape
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    if not leftx.size or not rightx.size:
        return None

    Lane.left.current_fit = np.polyfit(lefty, leftx, 2)
    Lane.right.current_fit = np.polyfit(righty, rightx, 2)

    Lane.left.hist_fit.append(Lane.left.current_fit)
    Lane.right.hist_fit.append(Lane.right.current_fit)

    find_best_fit()

    Lane.right.recent_xfitted = rightx
    Lane.left.recent_xfitted = leftx

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = Lane.left.best_fit[0] * ploty ** 2 + Lane.left.best_fit[1] * ploty + Lane.left.best_fit[2]
        right_fitx = Lane.right.best_fit[0] * ploty ** 2 + Lane.right.best_fit[1] * ploty + Lane.right.best_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    update_curvature(img_shape)
    Lane.car_position = find_car_position(img_shape)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result
