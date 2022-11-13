import math

import numpy as np
import cv2 as cv
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, dot, logpdf
from scipy.ndimage.filters import gaussian_filter


class LaneHistory:
    def __init__(self):
        self.left_fit = []
        self.right_fit = []
        self.left_curverad = []
        self.right_curverad = []
        self.radius = []
        self.lane_width = []
        self.center_off = []
        self.errorCount = 0
        self.steering_angle=0
        self.ploty = None

    def setWidth(self, value):
        self.lane_width = value
        return

    def setRadius(self, value):
        self.radius = value
        return

    def setPloty(self, value):
        self.ploty = value
        return

    def getError(self):
        return self.errorCount

    def setError(self, value):
        self.errorCount = value
        return

    def incrementError(self):
        self.errorCount += 1

    def GetLeft_fit(self):
        return self.left_fit

    def getDirection(self):
        return self.steering_angle

    def GetRight_fit(self):
        return self.right_fit

    def GetWidth(self):
        return self.lane_width[-1]

    def GetLeft_Curve(self):
        return self.left_curverad[-1]

    def GetRight_Curve(self):
        return self.right_curverad[-1]

    def GetCenterOff(self):
        return self.center_off[-1]

    def SetLeft_fit(self, left):
        self.left_fit.append(left)
        return

    def SetRight_fit(self, right):
        self.right_fit.append(right)
        return

    def SetLeftCurvature(self, leftCurve):
        self.left_curverad.append(leftCurve)
        return

    def SetRightCurvature(self, rightCurve):
        self.right_curverad.append(rightCurve)
        return

    def SetOffset(self, offset):
        self.center_off.append(offset)
        return

    def SetDirection(self,value):
        self.steering_angle = value

    def putHistoryDataOnScreen(self, img):
        # Show position

        lane_position_prcnt = self.GetCenterOff() / self.lane_width

        x_text_start, y_text_start = (10, 450)
        line_start = (10 + x_text_start, 40 + y_text_start)
        line_len = 300
        cv.putText(img, "Position", org=(x_text_start, y_text_start), fontScale=2, thickness=3,
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, lineType=cv.LINE_AA, color=(255, 255, 255))
        cv.line(img, color=(255, 255, 255), thickness=2,
                pt1=(line_start[0], line_start[1]),
                pt2=(line_start[0] + line_len, line_start[1]))
        cv.circle(img, center=(line_start[0] + int(lane_position_prcnt * line_len), line_start[1]),
                  radius=8,
                  color=(255, 255, 255))

        cv.putText(img, '{:.2f} m'.format(self.center_off[-1]), fontScale=1, thickness=1,
                   org=(line_start[0] + int(lane_position_prcnt * line_len) + 5, line_start[1] + 35),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), lineType=cv.LINE_AA)

        center = int(img.shape[1] / 2)
        cv.circle(img, center=(center, center + 5),
                  radius=8,
                  color=(255, 255, 255))

        # print to image
        text = "radius = %s [m]\nwidth = %s [m]\nLeft_curve = %s\nRight_curve = %s\nSteering_angle = %s [degree]" % (
            str(self.radius), str(self.lane_width), str(self.left_curverad[-1]),
            str(self.right_curverad[-1]), str(self.steering_angle))
        for i, line in enumerate(text.split('\n')):
            i = 50 + 20 * i
            cv.putText(img, line, (0, 200 + i), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        return


class Lane:
    def __init__(self, persp_t):
        self.image = []
        self.window_count = 9
        self.small_img_size = (256, 144)
        self.small_img_x_offset = 20
        self.small_img_y_offset = 10
        self.left_curverad = None
        self.right_curverad = None
        self.center_off = None
        self.lane_width = None
        self.radius = None
        self.left_fitx = None
        self.right_fitx = None
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 5.7 / 640  # meters per pixel in x dimension
        self.perspectiveT = persp_t
        self.canDraw = False
        self.left_kalmanFilter = WindowFilter(pos_init=1280 / 4)
        self.right_kalmanFilter = WindowFilter(pos_init=1280 - (1280 / 4))
        self.y_end = 500
        self.ploty = 600
        self.direction=None
        self.steering_angle=90
    def SetImg(self, img):
        self.image = img
        return

    def getRadius(self):
        return self.radius

    def getWidth(self):
        return self.lane_width

    def GetLeft_fitx(self):
        return self.left_fitx

    def GetRight_fitx(self):
        return self.right_fitx

    def GetLeft_Curve(self):
        return self.left_curverad

    def GetRight_Curve(self):
        return self.right_curverad

    def GetCenterOff(self):
        return self.center_off

    def getPloty(self):
        return self.ploty

    def getDirection(self):
        return self.steering_angle

    def putDatasOnScreen(self, img):

        # Show position

        lane_position_prcnt = self.center_off / self.lane_width

        x_text_start, y_text_start = (10, 450)
        line_start = (10 + x_text_start, 40 + y_text_start)
        line_len = 300
        cv.putText(img, "Position", org=(x_text_start, y_text_start), fontScale=2, thickness=3,
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, lineType=cv.LINE_AA, color=(255, 255, 255))
        cv.line(img, color=(255, 255, 255), thickness=2,
                pt1=(line_start[0], line_start[1]),
                pt2=(line_start[0] + line_len, line_start[1]))
        cv.circle(img, center=(line_start[0] + int(lane_position_prcnt * line_len), line_start[1]),
                  radius=8,
                  color=(255, 255, 255))
        cv.putText(img, '{:.2f} m'.format(self.center_off), fontScale=1, thickness=1,
                   org=(line_start[0] + int(lane_position_prcnt * line_len) + 5, line_start[1] + 35),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), lineType=cv.LINE_AA)

        # print to image

        center = int(self.image.shape[1] / 2)
        cv.circle(img, center=(center, center + 5),
                  radius=8,
                  color=(255, 255, 255))

        text = "radius = %s [m]\nwidth = %s [m]\nLeft_curve = %s\nRight_curve = %s\nSteering_angle = %s [degree]" % (
            str(self.radius), str(self.lane_width), str(self.left_curverad),
            str(self.right_curverad), str(self.steering_angle))
        for i, line in enumerate(text.split('\n')):
            i = 50 + 20 * i
            cv.putText(img, line, (0, 200 + i), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        return

    def fitPoly(self, left_lane_inds, right_lane_inds, left_end, right_end, nonzerox, nonzeroy):

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = []
        right_fit = []
        ploty = np.linspace(0, 600, self.image.shape[0])  # makes evenly spaced points of the lane points
        if left_end < right_end:
            self.y_end = left_end
        else:
            self.y_end = right_end
        self.ploty = ploty[self.y_end:]

        if len(leftx) > 0 and len(lefty) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            self.left_fitx = left_fit[0] * self.ploty ** 2 + left_fit[1] * self.ploty + left_fit[2]

        if len(rightx) > 0 and len(righty) > 0:
            right_fit = np.polyfit(righty, rightx, 2)
            self.right_fitx = right_fit[0] * self.ploty ** 2 + right_fit[1] * self.ploty + right_fit[2]

        if len(left_fit) > 0 and len(right_fit) > 0 and len(self.ploty) > 200:
            self.canDraw = True
            self.left_curverad, self.right_curverad, self.center_off = self.GetCurv(self.ploty, left_fit, right_fit)
        else:
            self.canDraw = False
        #print(str(self.left_fitx[0])+" "+str(self.right_fitx[0]))
        return left_fit, right_fit

    def validateWindow(self, nonzerox, good_left_inds, good_right_inds, rightx_current, leftx_current, win_y_high,
                       midpoint):
        # Set minimum number of pixels found to recenter window
        minpix = 200
        left_end = 100
        right_end = 100
        if len(good_left_inds) > minpix:
            self.left_kalmanFilter.update(rightx_current)
            lkf = int(self.left_kalmanFilter.get_position())

            if 200 < lkf < midpoint:
                leftx_current = lkf #- 25
            else:
                maxl = int(np.mean(nonzerox[good_left_inds]))
                avgl = int((lkf + maxl) / 2)
                leftx_current = avgl

            #leftx_current = int(np.mean(nonzerox[good_left_inds]))
            left_end = win_y_high
        if len(good_right_inds) > minpix:
            self.right_kalmanFilter.update(leftx_current)
            rkf = int(self.right_kalmanFilter.get_position())
            if midpoint < rkf < 860:
                rightx_current = rkf #+ 25
            else:
                maxr = int(np.mean(nonzerox[good_right_inds]))
                avgr = int((rkf + maxr) / 2)
                rightx_current = avgr
            #rightx_current = int(np.mean(nonzerox[good_right_inds]))
            right_end = win_y_high

        return rightx_current, leftx_current, left_end, right_end

    def sliding_windown(self, img_w, marginsize):
        """Returns the left and the right lane line points"""
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
        # Creates an output image to draw on and visualize the result
        out_img = np.dstack((img_w, img_w, img_w)) #* 255
        midpoint = int(histogram.shape[0] / 2)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Set height of windows
        window_height = int(img_w.shape[0] / self.window_count)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img_w.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        color_right = (0, 255, 0)  # green
        color_left = (0, 255, 0)  # green
        # if the base points of both lane lines are invalid, then make an approximation
        if leftx_current < 200:
            leftx_current = midpoint - 300
            color_left = (0, 255, 255)  # yellow
        if rightx_current > 1000:
            rightx_current = midpoint + 200
            color_right = (255, 255, 0)  # blue

            # self.prev_Left_Windows.append(leftx_current)
        self.left_kalmanFilter.set_init_poz(leftx_current)
        self.right_kalmanFilter.set_init_poz(rightx_current)

        right_end = self.image.shape[0]
        left_end = self.image.shape[0]

        # Set the width of the windows +/- margin
        margin = marginsize

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.window_count):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img_w.shape[0] - (window + 1) * window_height
            win_y_high = img_w.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color_left, 2)
            cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color_right, 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox <= win_xleft_high)).nonzero()[0]

            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox <= win_xright_high)).nonzero()[0]

            # determine the x position of the next window
            rightx_current, leftx_current, left_end, right_end = self.validateWindow(nonzerox, good_left_inds,
                                                                                     good_right_inds, rightx_current,
                                                                                     leftx_current, win_y_high,
                                                                                     midpoint)

            # Append these x,y koordinates to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Fit a second order polynomial to each
        left_fit, right_fit = self.fitPoly(left_lane_inds, right_lane_inds, left_end, right_end, nonzerox, nonzeroy)

        return left_fit, right_fit, out_img, left_lane_inds, right_lane_inds

    def draw_lines(self, img_w, perspective, color):
        """ Draws the lane to the original image"""
        # img =self.image
        # Creates an image to draw the lines on
        if self.canDraw:
            warp_zero = np.zeros_like(img_w).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            # Recast the x and y points into usable format for cv.fillPoly()

            pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])

            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv.fillPoly(color_warp, np.int_([pts]), color)

            # Warp the blank back to original image space using inverse perspective matrix
            newwarp = self.perspectiveT.perspective_transform(color_warp, perspective[1], perspective[0])

            # Combine the result with the original image
            result = cv.addWeighted(self.image, 1, newwarp, 0.2, 0)
            return result
        return self.image

    def draw_lines_fromHistory(self, img_w, left_fit, right_fit, ploty, perspective, color):
        """ Draws the lane to the original image"""
        # img =self.image
        # Creates an image to draw the lines on
        if len(left_fit) > 0 and len(right_fit) > 0:
            warp_zero = np.zeros_like(img_w).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]  # left polynom
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]  # right polynom

            # Recast the x and y points into usable format for cv.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv.fillPoly(color_warp, np.int_([pts]), color)

            # Warp the blank back to original image space using inverse perspective matrix
            newwarp = self.perspectiveT.perspective_transform(color_warp, perspective[1], perspective[0])

            # Combine the result with the original image
            result = cv.addWeighted(self.image, 1, newwarp, 0.2, 0)
            return result
        return self.image

    def display_heading_line(self,frame, steering_angle, line_color=(0, 0, 255), line_width=5):
        heading_image = np.zeros_like(frame)
        height, width, _ = frame.shape

        # figure out the heading line from steering angle
        # heading line (x1,y1) is always center bottom of the screen
        # (x2, y2) requires a bit of trigonometry

        # Note: the steering angle of:
        # 0-89 degree: turn left
        # 90 degree: going straight
        # 91-180 degree: turn right
        steering_angle_radian = steering_angle / 180.0 * math.pi
        x1 = int(width / 2)
        y1 = height-100
        x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
        y2 = int(height / 2)+20
        cv.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
        heading_image = cv.addWeighted(frame, 0.8, heading_image, 1, 1)

        return heading_image

    def stabilize_steering_angle(self,
            curr_steering_angle,
            new_steering_angle,
            max_angle_deviation_two_lines=3,):
        """
        Using last steering angle to stabilize the steering angle
        if new angle is too different from current angle,
        only turn by max_angle_deviation degrees
        """

        max_angle_deviation = max_angle_deviation_two_lines

        angle_deviation = new_steering_angle - curr_steering_angle
        if abs(angle_deviation) > max_angle_deviation:
            stabilized_steering_angle = int(curr_steering_angle
                                            + max_angle_deviation * angle_deviation / abs(angle_deviation))
        else:
            stabilized_steering_angle = new_steering_angle
        return stabilized_steering_angle
    def GetCurv(self, ploty, left_fit, right_fit):
        """Gets the curvature , radius, lane width, and center offset"""
        y_eval = np.max(ploty)  # Analyze only the top part of the lane
        center = int(self.image.shape[1] / 2)

        xLeft = self.left_fitx[0]
        xRight = self.right_fitx[0]

        xOffset = (xLeft+xRight)/2-center
        yOffset = int(self.image.shape[0])

        angle_to_radian = math.atan(xOffset / yOffset)
        angle_to_deg = int(angle_to_radian * 180.0 / math.pi)
        new_steering_angle= angle_to_deg + 90

        self.steering_angle = self.stabilize_steering_angle(self.steering_angle, new_steering_angle)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, self.left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, self.right_fitx * self.xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = round(((1 + (
                    2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0]),2)

        right_curverad = round(((1 + (
                    2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0]),2)

        # center = (right_fit[2] - left_fit[2]) / 2

        self.radius = round(right_curverad / left_curverad,2)

        self.lane_width = round(np.mean((self.right_fitx - self.left_fitx) * self.xm_per_pix),2)

        # center_off = round(center - self.image.shape[0] / 2. * self.xm_per_pix, 2)

        img_center = self.image.shape[1] / 2
        lane_position_prcnt = np.interp(img_center, [self.left_fitx[-1], self.right_fitx[-1]], [0, 1])


        lane_position = lane_position_prcnt * self.lane_width
        return right_curverad, left_curverad, lane_position

    def draw_lane_lines_regions(self, warped_img):
        """
        Returns an image where the computed left and right lane areas have been drawn on top of the original warped binary image
        """
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        margin = 50
        # ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        # ploty = ploty[self.y_end:]
        # Create RGB image from binary warped image
        region_img = np.dstack((warped_img, warped_img, warped_img)) #* 255

        if self.canDraw:
            left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx - margin, self.ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx + margin,
                                                                            self.ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))

            right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx - margin, self.ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx + margin,
                                                                             self.ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv.fillPoly(region_img, np.int_([left_line_pts]), (0, 255, 0))
            cv.fillPoly(region_img, np.int_([right_line_pts]), (0, 255, 0))

        return region_img

    def draw_lines_hotspots(self, warped_img, left_lane_inds, right_lane_inds):
        """
        Returns a RGB image where the portions of the lane lines that were
        identified by our pipeline are colored in blue (left) and red (right)
        """
        out_img = np.dstack((warped_img, warped_img, warped_img)) #* 255
        if self.canDraw:
            nonzero = warped_img.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            out_img[lefty, leftx] = [255, 255, 0]
            out_img[righty, rightx] = [0, 0, 255]

        return out_img

    def combine_images(self, orig_lane_img, lines_img, lines_regions_img, lane_hotspots_img, warped, rawLane):
        """
        Returns an image, where the lane detection steps are shown as in smaller windows on the original image
        """
        small_lines = cv.resize(lines_img, self.small_img_size)
        small_region = cv.resize(lines_regions_img, self.small_img_size)
        small_hotspots = cv.resize(lane_hotspots_img, self.small_img_size)
        warped_window = cv.resize(warped, self.small_img_size)
        raw_lane = cv.resize(rawLane, self.small_img_size)

        orig_lane_img[self.small_img_y_offset: self.small_img_y_offset + self.small_img_size[1],
        self.small_img_x_offset: self.small_img_x_offset + self.small_img_size[0]] = small_lines

        start_offset_y = self.small_img_y_offset
        start_offset_x = 2 * self.small_img_x_offset + self.small_img_size[0]

        orig_lane_img[start_offset_y: start_offset_y + self.small_img_size[1],
        start_offset_x: start_offset_x + self.small_img_size[0]] = small_region

        start_offset_y = self.small_img_y_offset
        start_offset_x = 3 * self.small_img_x_offset + 2 * self.small_img_size[0]

        orig_lane_img[start_offset_y: start_offset_y + self.small_img_size[1],
        start_offset_x: start_offset_x + self.small_img_size[0]] = small_hotspots

        start_offset_y = self.small_img_y_offset
        start_offset_x = 4 * self.small_img_x_offset + 3 * self.small_img_size[0]

        orig_lane_img[start_offset_y: start_offset_y + self.small_img_size[1],
        start_offset_x: start_offset_x + self.small_img_size[0]] = warped_window

        #start_offset_y = 500
        #start_offset_x = 10 * self.small_img_x_offset + 3 * self.small_img_size[0]

        #orig_lane_img[start_offset_y: start_offset_y + self.small_img_size[1],
        #start_offset_x: start_offset_x + self.small_img_size[0]] = raw_lane

        return orig_lane_img

    def sanity_check(self):
        """Decides whether the detected lane is valid or not"""
        #return True
        # print("---------------------------------------------------")
        if self.canDraw:
            #return True
            if self.lane_width > 3.8 or self.lane_width < 2.8:
                #print("LANE WIDTH FAIL: " +str(self.lane_width))
                return False
            if self.right_curverad < 300 or self.left_curverad < 300 or self.right_curverad > 15000 or self.left_curverad > 15000:
                #print("RIGHT CURVARAD FAIL: "+ str(self.right_curverad))
                #print("LEFT CURVARAD FAIL: " + str(self.left_curverad))
                return False
            if self.radius > 4 or self.radius < 0.2:
                #print("RADIUS FAIL: "  +str(self.radius))
                return False
            # if self.center_off<0:
            #    return  False
            return True
        else:
            # print("CANT DRAW")
            return False

    def region_of_interest(self, img):
        """Gets the ROI from the image"""
        mask = np.zeros_like(img)
        mask2 = np.zeros_like(img)
        imshape = img.shape

        vertices = np.array(
            [[(0, imshape[0] * .85), (imshape[1] * .30, imshape[0] * .45), (imshape[1] * .65, imshape[0] * .45),
              (imshape[1], imshape[0] * .85)]], dtype=np.int32)  # creates an array with the trapezoids verticies

        vertices4 = np.array(
            [[(0, imshape[0]), (imshape[1] * .38, imshape[0] * .65), (imshape[1] * .58, imshape[0] * .65),
              (imshape[1], imshape[0])]], dtype=np.int32)  # creates an array with the trapezoids verticies

        vertl = np.array([[(40, 720), (475, 375), (805, 375), (1180, 720)]], dtype=np.int32)
        vertm = np.array([[(190, 700), (580, 450), (730, 450), (1160, 700)]], dtype=np.int32)

        cv.fillPoly(mask, vertices, (255,) * 3)
        masked_image = cv.bitwise_and(img, mask)  # crops the original image with the mask

        # vert2 = np.array([[(300, imshape[0]), (imshape[1] * .52, imshape[0] * .58),
        #                     (imshape[1] * .7, imshape[0])]], dtype=np.int32)

        vert3 = np.array(
            [[(400, imshape[0]), (imshape[1] * .45, imshape[0] * .55), (imshape[1] * .47, imshape[0] * .55),
              (imshape[1] * .6, imshape[0])]], dtype=np.int32)

        cv.fillPoly(mask2, vert3, (255,) * 3)
        mask2 = cv.bitwise_not(mask2)

        masked_image_v = cv.bitwise_and(masked_image, mask2)
        return masked_image_v


class WindowFilter:
    def __init__(self, pos_init=0.0, meas_variance=40, process_variance=0.1, uncertainty_init=2 ** 10):
        """
        A one dimensional Kalman filter tuned to track the position of a window.
        State variable:   = [position,
                             velocity]
        :param pos_init: Initial position.
        :param meas_variance: Variance of each measurement. Decrease to have the filter chase each measurement faster.
        :param process_variance: Variance of each prediction. Decrease to follow predictions more.
        :param uncertainty_init: Uncertainty of initial position.
        """
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition matrix
        self.kf.F = np.array([[1., 1],
                              [0., 0.5]])

        # Measurement function
        self.kf.H = np.array([[1., 0.]])

        # Initial state estimate
        self.kf.x = np.array([pos_init, 0])

        # Initial Covariance matrix
        self.kf.P = np.eye(self.kf.dim_x) * uncertainty_init # updated the uncertanity

        # Measurement noise
        self.kf.R = np.array([[meas_variance]])

        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=process_variance)

    def update(self, pos):
        """
        Given an estimate x position, uses the kalman filter to estimate the most likely true position of the
        lane pixel.
        :param pos: measured x position of the pixel
        """
        self.kf.predict()
        self.kf.update(pos)


    def grow_uncertainty(self, mag):
        """Grows state uncertainty."""
        for i in range(mag):
            # P = FPF' + Q
            self.kf.P = self.kf._alpha_sq * dot(self.kf.F, self.kf.P, self.kf.F.T) + self.kf.Q

    def loglikelihood(self, pos):
        """Calculates the likelihood of a measurement given the filter parameters and gaussian assumption."""
        self.kf.S = dot(self.kf.H, self.kf.P, self.kf.H.T) + self.kf.R
        return logpdf(pos, np.dot(self.kf.H, self.kf.x), self.kf.S)

    def get_position(self):
        return self.kf.x[0]

    def set_init_poz(self, pos):
        self.kf.x = np.array([pos, 0])
