import numpy as np
import cv2 as cv
import matplotlib.pylab as plt
import glob
import pickle
import os

def calibrate_camera(image_files, nx, ny):
    objpoints = []
    imgpoints = []
    objp = np.zeros(shape=(nx * ny, 3), dtype=np.float32)
    gray = None
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    for i in image_files:
        img = cv.imread(i)
        if img.shape[0] != 720:
            img = cv.resize(img,(1280, 720))
        cv.imshow('image',img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (nx, ny))
        if ret:
            print("Calibrated!")
            imgpoints.append(corners)
            objpoints.append(objp)
    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def undistort_camera():
    out_examples = 0
    MOV_AVG_LENGTH = 5

    if not os.path.exists('calibration.p'):
        # Read all jpg files from calibration image folder
        images = glob.glob('camera_cal/*.jpg')
        with open('calibration.p', mode='wb') as f:
            ret, mtx, dist, rvecs, tvecs = calibrate_camera(images, nx=9, ny=6)
            pickle.dump([ret, mtx, dist, rvecs, tvecs], f)
            f.close()
    else:
        with open('calibration.p', mode='rb') as f:
            ret, mtx, dist, rvecs, tvecs = pickle.load(f)
            f.close()
    if out_examples:
        # output undistorted image to output_image
        to_calibrate = cv.imread('camera_cal/calibration3.jpg')
        cv.imsave('output_images/calibration3_calibrated.jpg', cv.undistort(to_calibrate, mtx, dist, None, mtx))
    return mtx, dist


def thresholding_pipeline(img, sobel_kernel=7, mag_thresh=(3, 255), s_thresh=(170, 255)):
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    gray = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Binary matrixes creation
    sobel_binary = np.zeros(shape=gray.shape, dtype=bool)
    s_binary = sobel_binary
    combined_binary = s_binary.astype(np.float32)
    # Sobel Transform
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = 0  # cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_abs = np.abs(sobelx ** 2 + sobely ** 2)
    sobel_abs = np.uint8(255 * sobel_abs / np.max(sobel_abs))
    sobel_binary[(sobel_abs > mag_thresh[0]) & (sobel_abs <= mag_thresh[1])] = 1
    # Threshold color channel
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Combine the two binary thresholds
    combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1
    combined_binary = np.uint8(255 * combined_binary / np.max(combined_binary))
    return combined_binary


def perspective_transform(img, src_m, dest_m):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_m)
    dest = np.float32(dest_m)
    M = cv.getPerspectiveTransform(src, dest)
    warped_img = cv.warpPerspective(img, M, img_size, flags=cv.INTER_LINEAR)
    return warped_img


def sliding_windown(img_w):
    histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img_w, img_w, img_w)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = int(img_w.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_w.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_w.shape[0] - (window + 1) * window_height
        win_y_high = img_w.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        recone = cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        rectwo = cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    # ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
    # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    #
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    return left_fit, right_fit


def roi(img, vert):
    # creates an array with zeros, which has the same shape like the img
    mask = np.zeros_like(img)

    match_mask_color = 255
    cv.fillPoly(mask, vert, match_mask_color)
    # reduce noise
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines):
    line_img = np.zeros((img.shape[0],
                         img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            # for x1, y1, x2, y2 in line:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
        img = cv.addWeighted(img, 0.8, line_img, 1, 1)  # draw lines to the original image
    return img


def make_coords(image, line_params):
    try:
        slope, intercept = line_params
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


# line : y=mx+b -> m=slope, b=intercept
# we can avarage the m,b values to a single line
# first we have to separate the coordinates, which belongs to the left and right line
def avarage_slope_intercept(image, lines):
    left_fit = []  # coordinates of the left line
    right_fit = []  # coordinates of the right line
    if lines is not None:
        for line in lines:  # loop through the lines
            x1, y1, x2, y2 = line.reshape(4)  # endpoints of a line
            params = np.polyfit((x1, x2), (y1, y2), 1)  # makes a polynomial fuction which from the endpoints
            slope = params[0]
            intercept = params[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coords(image, left_fit_average)
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coords(image, right_fit_average)

        return np.array([left_line, right_line])


def process(image):
    # shape of the image
    # print(image.shape)

    # define height and width
    height = image.shape[0]
    width = image.shape[1]

    # grayscale
    gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # Define ROI

    ROI_vert = [(0, height), (1200, height), (550, 300)]  # test2.mp4
    # ROI_vert = [(0, 650), (1200, height), (700,250)] #lanetest.mp4
    canny_image = cv.Canny(gray_img, 10, 200)
    cropped_img = roi(thresholding_pipeline(image), np.array([ROI_vert], np.int32))

    # plt.imshow(cropped_img)
    # plt.show()

    lines = cv.HoughLinesP(cropped_img, rho=6, theta=np.pi / 180, threshold=140, lines=np.array([]),
                           minLineLength=40, maxLineGap=60)

    avg_lines = avarage_slope_intercept(image, lines)

    image_lines = draw_lines(image, avg_lines)
    return image_lines


def region_of_interest(img):
    mask = np.zeros_like(img)

    imshape = img.shape
    if len(imshape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    vertices = np.array([[(0, imshape[0]), (imshape[1] * .48, imshape[0] * .58), (imshape[1] * .52, imshape[0] * .58),(imshape[1], imshape[0])]], dtype=np.int32)
    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image


def _createSource():
    _imageSize = (1280, 720)
    xOffsetBottom = 200
    xOffsetMiddle = 595
    yOffset = 450
    sourceBottomLeft = (xOffsetBottom, _imageSize[1])
    sourceBottomRight = (_imageSize[0] - xOffsetBottom, _imageSize[1])
    sourceTopLeft = (xOffsetMiddle, yOffset)
    sourceTopRight = (_imageSize[0] - xOffsetMiddle, yOffset)
    return np.float32([sourceBottomLeft, sourceTopLeft, sourceTopRight, sourceBottomRight])


def _createDestination():
    _imageSize = (1280, 720)
    xOffset = _imageSize[0] / 4
    destinationBottomLeft = (xOffset, _imageSize[1])
    destinationBottomRight = (_imageSize[0] - xOffset, _imageSize[1])
    destinationTopLeft = (xOffset, 0)
    destinationTopRight = (_imageSize[0] - xOffset, 0)
    return np.float32([destinationBottomLeft, destinationTopLeft, destinationTopRight, destinationBottomRight])


def draw_lines(img, img_w, left_fit, right_fit, perspective):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_w).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # color_warp_center = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    # cv2.fillPoly(color_warp_center, np.int_([pts]), (0, 255, 0))
    cv.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_transform(color_warp, perspective[1], perspective[0])
    # Combine the result with the original image
    result = cv.addWeighted(img, 1, newwarp, 0.2, 0)
    color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
    cv.polylines(color_warp_lines, np.int_([pts_right]), isClosed=False, color=(255, 255, 0), thickness=25)
    cv.polylines(color_warp_lines, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=25)
    newwarp_lines = perspective_transform(color_warp_lines, perspective[1], perspective[0])
    result = cv.addWeighted(result, 1, newwarp_lines, 1, 0)
    return result


def process_adv(image):
    height = image.shape[0]
    width = image.shape[1]
    top_left = [300, 300]
    top_right = [800, 300]
    bottom_right = [1200, height]
    bottom_left = [0, height]

    src_mask = np.array([[(top_left[0], top_left[1]), (top_right[0], top_right[1]),
                          (bottom_right[0], bottom_right[1]), (bottom_left[0], bottom_left[1])]], np.int32)
    dst_mask = np.array([[(bottom_left[0], 0), (bottom_right[0], 0),
                          (bottom_right[0], bottom_right[1]), (bottom_left[0], bottom_left[1])]], np.int32)

    dest_mask = _createDestination()
    s_mask = _createSource()

    combined_img = thresholding_pipeline(image)
    roi_image = region_of_interest(combined_img)

    warped = perspective_transform(combined_img, s_mask, dest_mask)

    left_fit, right_fit = sliding_windown(warped)
    result = draw_lines(image, warped, left_fit, right_fit, perspective=[s_mask, dest_mask])
    # result = project_lanelines(warped, image, left_fit, right_fit, dest_mask, s_mask)

    return result


capture = cv.VideoCapture('project_video.mp4')

mtx, dist = undistort_camera()

while capture.isOpened():
    ret, frame = capture.read()
    # if frame is read correctly ret is True
    img = cv.undistort(cv.cvtColor(frame, cv.COLOR_RGB2BGR), mtx, dist, None, mtx)
    frame = process_adv(img)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
