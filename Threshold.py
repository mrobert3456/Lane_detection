import cv2 as cv
import numpy as np

class ImgThreshold:


    def thresholding_pipeline(self,image,sobel_kernel=7, mag_thresh=(3, 255), s_thresh=(170, 255), mod="HSV"):
        hsv_image = image
        if mod == "HSV":
            hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)  # converts the input image into hsv colour space
            gray = hsv_image[:, :, 2]  # gets the grayscale image
        elif mod == "LAB":
            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2Lab)  # converts the input image into hls colour space
            gray = hsv_image[:, :, 0]  # gets the grayscale image
        elif mod =="HLS":
            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HLS)  # converts the input image into hls colour space
            gray = hsv_image[:, :, 2]  # gets the grayscale image

        s_channel = hsv_image[:, :, 2]  # gets the saturation of the image

        sobel_zero = np.zeros(shape=gray.shape,
                              dtype=bool)  # creates an image with the same shape as grayscale image and fills it with zeros
        hls_zero = sobel_zero
        combined_binary = hls_zero.astype(np.float32)  # converts the s_binary to float32

        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)  # computes the x partial derivatives

        sobel_abs = np.abs(sobelx ** 2)  # takes the absolute value of sobelx^2

        sobel_abs = np.uint8(
            255 * sobel_abs / np.max(
                sobel_abs))  # scale it to 8 bit then converts to uint8, it would be noisy without it

        sobel_zero[(sobel_abs > mag_thresh[0]) & (
                sobel_abs <= mag_thresh[1])] = 1  # writes 1 where the the values is in the magnitude threshold

        hls_zero[(s_channel >= s_thresh[0]) & (
                s_channel <= s_thresh[1])] = 1  # writes 1 where the the values is in the s_channel threshold

        # Combines the two thresholds
        combined_binary[(hls_zero == 1) | (sobel_zero == 1)] = 1

        combined_binary = np.uint8(
            255 * combined_binary / np.max(combined_binary))  # scale it to 8 bit then converts to uint8

        return combined_binary

    def img_thres(self,img):
        """
                Takes a road image and returns an image where pixel intensity maps to likelihood of it being part of the lane.
                Each pixel gets its own score, stored as pixel intensity. An intensity of zero means it is not from the lane,
                and a higher score means higher confidence of being from the lane.
                :param img: an image of a road, typically from an overhead perspective.
                :return: The score image.
                """
        # Settings to run thresholding operations on
        settings = [ #{'name': 'lab_b', 'cspace': 'LAB', 'channel': 2, 'clipLimit': 1.0, 'threshold': 180},,
                    {'name': 'lightness', 'cspace': 'HLS', 'channel': 1, 'clipLimit': 1.0, 'threshold': 210}]#,
                    #{'name': 'value', 'cspace': 'HSV', 'channel': 2, 'clipLimit': 1.0, 'threshold': 220}]


        # Perform binary thresholding according to each setting and combine them into one image.
        scores = np.zeros(img.shape[0:2]).astype('uint8')
        for params in settings:
            # Change color space
            color_t = getattr(cv, 'COLOR_RGB2{}'.format(params['cspace']))
            gray = cv.cvtColor(img, color_t)[:, :, params['channel']]

            # Normalize regions of the image using CLAHE
            clahe = cv.createCLAHE(params['clipLimit'], tileGridSize=(8, 8))
            norm_img = clahe.apply(gray)

            # Threshold to binary
            ret, binary = cv.threshold(norm_img, params['threshold'], 1, cv.THRESH_BINARY)

            scores += binary

            # Save images
            #self.viz_save(params['name'], gray)
            #self.viz_save(params['name'] + '_binary', binary)

        res= cv.normalize(scores, None, 0, 255, cv.NORM_MINMAX)
        #canny =  cannyimg = cv.Canny(res, 150, 200)

        return res