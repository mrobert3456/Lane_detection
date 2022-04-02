
import cv2 as cv
from timeit import default_timer as timer
from Lane import Lane
from Lane import LaneHistory
from Perspective_transform import Perspective
from Threshold import ImgThreshold
from TrafficSign_recognition import TrafficSignDetector
import numpy as np

perspective =Perspective()
laneHistory =LaneHistory()
laneProcess = Lane(perspective)
Thresholder =ImgThreshold()
SignDetector = TrafficSignDetector()
GoodLane = True
capture = cv.VideoCapture('higwaytest.mp4')
# FPS counter
counter = 0
fps_start = timer()
firstLane=True
def Process_adv(image):

    laneProcess.SetImg(image)
    perspective.setImg(image)

    dest_mask = perspective.setDestination()
    s_mask = perspective.setSource()

    #combined_img = Thresholder.thresholding_pipeline(image)

    #get ROI
    roi_image = laneProcess.region_of_interest(image)
    #return roi_image

    #binaryze the image
    combined_img = Thresholder.img_thres(roi_image)
    #return  combined_img

    #Perspective transform
    warped = perspective.perspective_transform(combined_img, s_mask, dest_mask)

    #return warped

    #blurred = cv.GaussianBlur(warped,(3,3),5)
    #return blurred
    global GoodLane
    global  firstLane

    #if the lane is good then the marginsize =50, else marginsize=100

    if GoodLane:
        left_fit, right_fit, outimg,left_lane_inds, right_lane_inds = laneProcess.sliding_windown(warped,marginsize=25)  # returns the right and the left lane lines points
    else:
        left_fit, right_fit, outimg,left_lane_inds, right_lane_inds = laneProcess.sliding_windown(warped, marginsize=50)  # returns the right and the left lane lines points
    #return outimg


    drawn_lines_regions = laneProcess.draw_lane_lines_regions(warped)
    #cv.imshow('b',drawn_lines_regions)
    #return drawn_lines_regions
    drawn_hotspots = laneProcess.draw_lines_hotspots(warped, left_lane_inds, right_lane_inds)
    #cv.imshow('c', drawn_hotspots)
    #return drawn_hotspots
    validLane = laneProcess.sanity_check()
    if  validLane or firstLane:
        result = laneProcess.draw_lines(warped, perspective=[s_mask, dest_mask], color=(0, 255, 0))
        if validLane:
            GoodLane = True
            laneHistory.setError(0)
            laneHistory.SetLeft_fit(left_fit)  # good lanes are added to the history
            laneHistory.SetRight_fit(right_fit)
            laneHistory.SetLeftCurvature(laneProcess.GetLeft_Curve())
            laneHistory.SetRightCurvature(laneProcess.GetRight_Curve())
            laneHistory.SetOffset(laneProcess.GetCenterOff())
            laneHistory.setWidth(laneProcess.getWidth())
            laneHistory.setRadius(laneProcess.getRadius())
            laneHistory.setPloty(laneProcess.getPloty())
            laneProcess.putDatasOnScreen(result)
        firstLane=False

    else:
        GoodLane = False

        if laneHistory.getError()>20:
            laneHistory.setError(0)
            firstLane=True
            prev_l =laneHistory.GetLeft_fit()[-1]
            prev_r =laneHistory.GetRight_fit()[-1]
            laneProcess.canDraw = False
            laneHistory.GetLeft_fit().clear()
            laneHistory.GetRight_fit().clear()

            #laneHistory.SetLeft_fit(prev_l)
            #laneHistory.SetRight_fit(prev_r)

        if len(laneHistory.GetLeft_fit()) >0 and len(laneHistory.GetRight_fit())>0:
            #avg_left_fitx, avg_right_fitx = get_avg_lane()
            result = laneProcess.draw_lines_fromHistory( warped,  laneHistory.GetLeft_fit()[-1], laneHistory.GetRight_fit()[-1],laneHistory.ploty, perspective=[s_mask, dest_mask],color=(0, 255, 255))
            laneHistory.incrementError()
            laneHistory.putHistoryDataOnScreen(result)
        else:
            result = image
    roi_og = laneProcess.region_of_interest(image)
    warped_or = perspective.perspective_transform(roi_og, s_mask, dest_mask)
    laneProcess.combine_images(result, outimg, drawn_lines_regions, drawn_hotspots, warped_or)
    laneProcess.canDraw=False
    return result

def get_avg_lane():
    """Gets the avarage of the detected lanes"""
    if len(laneHistory.GetLeft_fit()) < 2:  # if the history size is less than 2 ,then returns the latest history
        return laneHistory.GetLeft_fit()[-1], laneHistory.GetRight_fit()[-1]

    left_avg = laneHistory.GetLeft_fit()[-1]
    right_avg = laneHistory.GetRight_fit()[-1]
    n_lanes = len(laneHistory.GetLeft_fit())

    for i in range(0, n_lanes):
        left_avg = np.add(left_avg, laneHistory.GetLeft_fit()[i])
        right_avg = np.add(right_avg, laneHistory.GetRight_fit()[i])

    avg_left_fitx = left_avg / n_lanes
    avg_right_fitx = right_avg / n_lanes

    return avg_left_fitx, avg_right_fitx


while capture.isOpened():
    ret, frame1 = capture.read()
    ret2, frame = capture.read()
    # if frame is read correctly ret is True
    frame =cv.resize(frame,(1280,720))
    #frame =SignDetector.DetectSign(frame)
    frame = Process_adv(frame)

    cv.imshow('frame', frame)

    #print("frame:"+str(fdb))
    #fdb+=1

    counter += 2

    # Stopping timer for FPS
    # Getting current time point in seconds
    fps_stop = timer()

    # Checking if timer reached 1 second
    # Comparing
    if fps_stop - fps_start >= 1.0:
        # Showing FPS rate
        print('FPS rate is: ', counter)
        # Reset FPS counter
        counter = 0
        # Restart timer for FPS
        # Getting current time point in seconds
        fps_start = timer()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()