
import numpy as np
import cv2 as cv
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter, dot, logpdf
from scipy.ndimage.filters import gaussian_filter

class LaneHistory:
    def __init__(self):
        self.left_fit = []
        self.right_fit = []
        self.left_curverad=[]
        self.right_curverad=[]
        self.radius =[]
        self.lane_width=[]
        self.center_off=[]
        self.errorCount=0


    def setWidth(self,value):
        self.lane_width=value
        return
    def setRadius(self,value):
        self.radius=value
        return

    def getError(self):
        return self.errorCount
    def setError(self,value):
        self.errorCount=value
        return
    def incrementError(self):
        self.errorCount+=1

    def GetLeft_fit(self):
        return self.left_fit

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

    def putHistoryDataOnScreen(self, img):
        # print to image
        text = "radius = %s [m]\noffcenter = %s [m]\nwidth = %s [m]\nLeft_curve = %s\nRight_curve = %s" % (
            str(self.radius), str(self.center_off[-1]), str(self.lane_width), str(self.left_curverad[-1]),
            str(self.right_curverad[-1]))
        for i, line in enumerate(text.split('\n')):
            i = 50 + 20 * i
            cv.putText(img, line, (0, 200 + i), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        return

class Lane:
    def __init__(self,persp_t):
        self.image= []
        self.window_count=15
        self.small_img_size = (256, 144)
        self.small_img_x_offset = 20
        self.small_img_y_offset = 10
        #self.imgWidth=binImg.shape[1]
        #self.imgHeight=binImg.shape[0]
        self.left_curverad = None
        self.right_curverad = None
        self.center_off=None
        self.lane_width=None
        self.radius=None
        self.left_fitx=None
        self.right_fitx=None
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 640  # meters per pixel in x dimension
        self.perspectiveT=persp_t
        self.filter = WindowFilter(pos_init=1280 / 4)
        self.prev_Left_Windows=[]
        self.prev_Right_Windows=[]
        self.dropout_Count=0
        self.leftDiff=0
        self.rightDiff=0

        self.canDraw=False
        self.left_kalmanFilter = WindowFilter()
        self.right_kalmanFilter = WindowFilter()



    def SetImg(self,img):
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


    def putDatasOnScreen(self, img):

        # print to image
        text = "radius = %s [m]\noffcenter = %s [m]\nwidth = %s [m]\nLeft_curve = %s\nRight_curve = %s" % (
            str(self.radius), str(self.center_off), str(self.lane_width), str(self.left_curverad), str(self.right_curverad))
        for i, line in enumerate(text.split('\n')):
            i = 50 + 20 * i
            cv.putText(img, line, (0, 200 + i), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        return


    def sliding_windown(self,img_w,marginsize):
        """Returns the left and the right lane line points"""
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
        # histogram = parallel.HistogramGPU(img_w) # gets the histogram of the image

        # Creates an output image to draw on and visualize the result
        out_img = np.dstack((img_w, img_w, img_w)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0] / 2)
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

        if leftx_current==0:
            leftx_current = midpoint-250

        self.prev_Left_Windows.append(leftx_current)
        self.left_kalmanFilter.set_init_poz(leftx_base)
        self.right_kalmanFilter.set_init_poz(rightx_base)
        min_lanepoint_goodness = -40

        # Set the width of the windows +/- margin
        margin = marginsize

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        color = (0, 255, 0)  # green
        db=0
        # Step through the windows one by one
        for window in range(self.window_count):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img_w.shape[0] - (window + 1) * window_height
            win_y_high = img_w.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # indentify whether the left and the right windows are intersected

            # Draw the windows on the visualization image

            recone = cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), color, 2)
            rectwo = cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), color, 2)

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
                #leftx_current = int(np.mean(nonzerox[good_left_inds]))
                self.left_kalmanFilter.update(rightx_current)
                seged =int(self.left_kalmanFilter.get_position())
                if seged >0:
                    leftx_current=seged
                else:
                    leftx_current =int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                #rightx_current = int(np.mean(nonzerox[good_right_inds]))
                self.right_kalmanFilter.update(leftx_current)
                seged =int(self.right_kalmanFilter.get_position())

                if seged >0:
                    rightx_current=seged
                else:
                    rightx_current =int(np.mean(nonzerox[good_left_inds]))




        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        #use the other correctly detected lane line
        #if len(leftx)<200 and len(rightx)>100:
        #    leftx = np.zeros_like(rightx)
        #    kul = 1280-rightx
        #    leftx+=kul-130
        #    lefty=righty

        #if len(rightx) < 200 and len(leftx) > 100:
        #    rightx = np.zeros_like(leftx)
        #    kul = 1280 - leftx
        #    rightx += kul -100
        #    righty = lefty



        #difflx= np.diff(leftx)
        #diffly = np.diff(lefty)

        #diffrx =np.diff(rightx)
        #diffry =np.diff(righty)

        #leftx,lefty =self.checkPoints(leftx,lefty,difflx,diffly)
        #rightx,righty =self.checkPoints(rightx,righty,diffrx,diffry)

        # Fit a second order polynomial to each
        left_fit =[]
        right_fit=[]
        ploty = np.linspace(0, self.image.shape[0] - 1,self.image.shape[0])  # makes evenly spaced points of the lane points
        if len(leftx)>0 or len(lefty)>0 :
            left_fit = np.polyfit(lefty, leftx, 2)
            self.left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]


        if len(rightx) > 0 or len(righty) > 0:
            right_fit = np.polyfit(righty, rightx, 2)
            self.right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        if len(left_fit)>0 and len(right_fit):
            self.canDraw = True
            self.left_curverad,self.right_curverad,self.center_off = self.GetCurv(ploty,left_fit,right_fit)

        return left_fit, right_fit, out_img, left_lane_inds, right_lane_inds

    def checkPoints(self,xkoords,ykoords,diffx,diffy):

        for i in range(0,len(diffx)):
            if diffx[i]>10:
                np.delete(xkoords,i+1)
                np.delete(ykoords,i+1)
            if diffx[i]<-10:
                np.delete(xkoords, i )
                np.delete(ykoords, i )

        return xkoords,ykoords
    def draw_lines(self,img_w,perspective,color):
        """ Draws the lane to the original image"""
        #img =self.image
        # Creates an image to draw the lines on
        if self.canDraw:
            warp_zero = np.zeros_like(img_w).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # Find the inflection point of the lane, so the polynom can be corrected

            #diff2_l = np.gradient(np.gradient(self.left_fit, 1), 1)  # gets the second derevative to determine inflection point
            #diff2_r = np.gradient(np.gradient(self.right_fit, 1), 1)  # gets the second derevative to determine inflection point

            ploty = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])  # makes evenly spaced points of the lane points

            diff_left = np.gradient(self.left_fitx, 1)
            sdiff_left = np.gradient(np.gradient(diff_left, 1), 1)

            diff_right = np.gradient(self.right_fitx, 1)
            sdiff_right = np.gradient(np.gradient(diff_right, 1), 1)

            infl_left = -1
            infl_right = -1

            positive = diff_left[0] > 0 and diff_right[0] > 0

            ploty = ploty[:600]
            self.left_fitx = self.left_fitx[:600]
            self.right_fitx = self.right_fitx[:600]

            # Recast the x and y points into usable format for cv.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([self.left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, ploty])))])

            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv.fillPoly(color_warp, np.int_([pts]), color)

            # Warp the blank back to original image space using inverse perspective matrix
            newwarp = self.perspectiveT.perspective_transform(color_warp, perspective[1], perspective[0])

            # Combine the result with the original image
            result = cv.addWeighted(self.image, 1, newwarp, 0.2, 0)
            return result
        return self.image

    def draw_lines_fromHistory(self,img_w, left_fit, right_fit, perspective,color):
        """ Draws the lane to the original image"""
        # img =self.image
        # Creates an image to draw the lines on
        if len(left_fit)>0 and len(right_fit)>0:
            warp_zero = np.zeros_like(img_w).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

            # Find the inflection point of the lane, so the polynom can be corrected

            #diff2_l = np.gradient(np.gradient(left_fit, 1), 1)  # gets the second derevative to determine inflection point
            #diff2_r = np.gradient(np.gradient(right_fit, 1), 1)  # gets the second derevative to determine inflection point

            ploty = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])  # makes evenly spaced points of the lane points


            # ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])  # makes evenly spaced points of the lane points
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]  # left polynom
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]  # right polynom

            diff_left = np.gradient(left_fitx, 1)
            #sdiff_left = np.gradient(np.gradient(diff_left, 1), 1)

            diff_right = np.gradient(right_fitx, 1)
            #sdiff_right = np.gradient(np.gradient(diff_right, 1), 1)

            infl_left = -1
            infl_right = -1

            positive = diff_left[0] > 0 and diff_right[0] > 0

            ploty = ploty[:600]
            left_fitx = left_fitx[:600]
            right_fitx = right_fitx[:600]


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
    def GetCurv(self,ploty, left_fit, right_fit):
        """Gets the curvature , radius, lane width, and center offset"""
        y_eval = np.max(ploty)  # Analyze only the top part of the lane

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, self.left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, self.right_fitx * self.xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])

        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        center = (right_fit[2] - left_fit[2]) / 2

        #left_off = (center - left_fit[2]) * self.xm_per_pix
        #right_off = (right_fit[2] - center) * self.xm_per_pix

        self.radius = right_curverad / left_curverad

        # lane_width = (right_fit[2] - left_fit[2]) * xm_per_pix
        self.lane_width = np.mean((self.right_fitx - self.left_fitx) * self.xm_per_pix)

        center_off = round(center - self.image.shape[0] / 2. * self.xm_per_pix, 2)

        return right_curverad, left_curverad, center_off

    def draw_lane_lines_regions(self,warped_img):
        """
        Returns an image where the computed left and right lane areas have been drawn on top of the original warped binary image
        """
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        margin = 50
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])
        # Create RGB image from binary warped image
        region_img = np.dstack((warped_img, warped_img, warped_img)) * 255

        if self.canDraw:
            left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx + margin,
                                                                            ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))

            right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx + margin,
                                                                             ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv.fillPoly(region_img, np.int_([left_line_pts]), (0, 255, 0))
            cv.fillPoly(region_img, np.int_([right_line_pts]), (0, 255, 0))

        return region_img

    def draw_lines_hotspots(self,warped_img, left_lane_inds, right_lane_inds):
        """
        Returns a RGB image where the portions of the lane lines that were
        identified by our pipeline are colored in blue (left) and red (right)
        """
        out_img = np.dstack((warped_img, warped_img, warped_img)) * 255
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

    def combine_images(self,orig_lane_img, lines_img, lines_regions_img, lane_hotspots_img, warped):
        """
        Returns an image, where the lane detection steps are shown as in smaller windows on the original image
        """
        small_lines = cv.resize(lines_img, self.small_img_size)
        small_region = cv.resize(lines_regions_img, self.small_img_size)
        small_hotspots = cv.resize(lane_hotspots_img, self.small_img_size)
        warped_window = cv.resize(warped, self.small_img_size)

        orig_lane_img[self.small_img_y_offset: self.small_img_y_offset + self.small_img_size[1],
        self.small_img_x_offset: self.small_img_x_offset + self.small_img_size[0]] = small_lines

        start_offset_y = self.small_img_y_offset
        start_offset_x = 2 * self.small_img_x_offset + self.small_img_size[0]

        orig_lane_img[start_offset_y: start_offset_y + self.small_img_size[1],start_offset_x: start_offset_x + self.small_img_size[0]] = small_region

        start_offset_y = self.small_img_y_offset
        start_offset_x = 3 * self.small_img_x_offset + 2 * self.small_img_size[0]

        orig_lane_img[start_offset_y: start_offset_y + self.small_img_size[1],
        start_offset_x: start_offset_x + self.small_img_size[0]] = small_hotspots

        start_offset_y = self.small_img_y_offset
        start_offset_x = 4 * self.small_img_x_offset + 3 * self.small_img_size[0]

        orig_lane_img[start_offset_y: start_offset_y + self.small_img_size[1],
        start_offset_x: start_offset_x + self.small_img_size[0]] = warped_window

        return orig_lane_img
    def sanity_check(self):
        """Decides whether the detected lane is valid or not"""
        #return True
        if self.canDraw:
            if self.lane_width > 3.4 or self.lane_width < 2.0:
                return False
            #if self.right_curverad < 1000 and self.left_curverad < 1000 and self.right_curverad > 100 and self.left_curverad > 100:
            if self.radius > 2 or self.radius < 0.2:
                return False
            return True
        else:
            return False

    def region_of_interest(self, img):
        """Gets the ROI from the image"""
        mask = np.zeros_like(img)
        mask2 = np.zeros_like(img)
        imshape = img.shape

        vertices = np.array(
             [[(0, imshape[0]*.85), (imshape[1] * .40, imshape[0] * .45), (imshape[1] * .48, imshape[0] * .45),
               (imshape[1], imshape[0]*.85)]], dtype=np.int32)  # creates an array with the trapezoids verticies

        vertices4 = np.array(
            [[(0, imshape[0]), (imshape[1] * .38, imshape[0] * .65), (imshape[1] * .58, imshape[0] * .65),
              (imshape[1], imshape[0])]], dtype=np.int32)  # creates an array with the trapezoids verticies


        cv.fillPoly(mask, vertices, (255,) * 3)
        masked_image = cv.bitwise_and(img, mask)  # crops the original image with the mask

        # vert2 = np.array([[(300, imshape[0]), (imshape[1] * .52, imshape[0] * .58),
        #                     (imshape[1] * .7, imshape[0])]], dtype=np.int32)

        vert3 = np.array(
            [[(300, imshape[0]), (imshape[1] * .45, imshape[0] * .65), (imshape[1] * .52, imshape[0] * .65),
              (imshape[1] * .7, imshape[0])]], dtype=np.int32)

        cv.fillPoly(mask2, vert3, (255,) * 3)
        mask2 = cv.bitwise_not(mask2)

        masked_image_v = cv.bitwise_and(masked_image, mask2)
        return masked_image_v



class WindowFilter:
    def __init__(self, pos_init=0.0, meas_variance=100, process_variance=0.1, uncertainty_init=2 ** 30):
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

        # State transition function
        self.kf.F = np.array([[1., 1],
                              [0., 0.5]])

        # Measurement function
        self.kf.H = np.array([[1., 0.]])

        # Initial state estimate
        self.kf.x = np.array([pos_init, 0])

        # Initial Covariance matrix
        self.kf.P = np.eye(self.kf.dim_x) * uncertainty_init

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