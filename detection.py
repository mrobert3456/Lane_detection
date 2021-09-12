import numpy as np
import cv2 as cv
import matplotlib.pylab as plt


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


def avarage_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            params = np.polyfit((x1, x2), (y1, y2), 1)
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
    cropped_img = roi(canny_image, np.array([ROI_vert], np.int32))

    # plt.imshow(cropped_img)
    # plt.show()

    lines = cv.HoughLinesP(cropped_img, rho=6, theta=np.pi / 180, threshold=140, lines=np.array([]),
                           minLineLength=40, maxLineGap=60)

    avg_lines = avarage_slope_intercept(image, lines)

    image_lines = draw_lines(image, avg_lines)
    return image_lines


capture = cv.VideoCapture('test2.mp4')

while capture.isOpened():
    ret, frame = capture.read()
    # if frame is read correctly ret is True
    frame = process(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
