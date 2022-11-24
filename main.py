
import cv2 as cv
from timeit import default_timer as timer
from Lane import Lane
from Lane import LaneHistory
from Perspective_transform import Perspective
from Threshold import ImgThreshold
from TrafficSign_recognition import TrafficSignDetector
from tqdm import tqdm
import time
import numpy as np

perspective =Perspective()
laneHistory =LaneHistory()
laneProcess = Lane(perspective)
Thresholder =ImgThreshold()
SignDetector = TrafficSignDetector()
GoodLane = True
firstLane=True
capture = cv.VideoCapture('higwaytest.mp4')
# FPS counter
counter = 0
fps_start = timer()


def Detect_lane(image):

    laneProcess.SetImg(image)
    perspective.setImg(image)

    dest_mask = perspective.setDestination()
    s_mask = perspective.setSource()

    #get ROI
    roi_image = laneProcess.regionOfInterest(image)

    #binaryze the image
    combined_img = Thresholder.img_thres(roi_image)

    #Perspective transform
    warped = perspective.perspective_transform(combined_img, s_mask, dest_mask)

    global GoodLane
    global firstLane


    #if the lane is good then the marginsize =50, else marginsize=100

    if GoodLane:
        left_fit, right_fit, outimg,left_lane_inds, right_lane_inds = laneProcess.slidingWindown(warped,marginsize=25)  # returns the right and the left lane lines points
    else:
        left_fit, right_fit, outimg,left_lane_inds, right_lane_inds = laneProcess.slidingWindown(warped, marginsize=50)  # returns the right and the left lane lines points

    # draw detection steps results to the input frame
    drawn_lines_regions = laneProcess.drawLaneLinesRegions(warped)

    drawn_hotspots = laneProcess.drawLinesHotspots(warped, left_lane_inds, right_lane_inds)

    raw_lane = laneProcess.drawLines(warped, perspective=[s_mask, dest_mask], color=(0, 255, 0))

    #sanity check
    validLane = laneProcess.sanityCheck()
    if validLane or firstLane:
        result = raw_lane
        if validLane: # good lanes are added to the history
            GoodLane = True
            laneHistory.setHistory(0, left_fit, right_fit, laneProcess.getLeftCurve(), laneProcess.getRightCurve(),
                                   laneProcess.getCenterOff(), laneProcess.getWidth(), laneProcess.getRadius(),
                                   laneProcess.getPloty(), laneProcess.getDirection())
            laneProcess.putDatasOnScreen(result)
            result = laneProcess.displayHeadingLine(result,laneProcess.getDirection())
        firstLane=False

    else:
        GoodLane = False
        # if detection error count is above 20
        if laneHistory.getError()>20:
            laneHistory.setError(0)
            firstLane=True
            laneProcess.canDraw = False
            laneHistory.getLeftFit().clear()
            laneHistory.getRightFit().clear()

        #if history has data
        if len(laneHistory.getLeftFit()) >0 and len(laneHistory.getRightFit())>0:
            result = laneProcess.drawLinesFromHistory( warped,  laneHistory.getLeftFit()[-1], laneHistory.getRightFit()[-1],laneHistory.ploty, perspective=[s_mask, dest_mask],color=(0, 255, 255))
            laneHistory.incrementError()
            result  = laneProcess.displayHeadingLine(result,laneHistory.getDirection())
            laneHistory.putHistoryDataOnScreen(result)
        else:
            result = image
    roi_og = laneProcess.regionOfInterest(image)
    warped_or = perspective.perspective_transform(roi_og, s_mask, dest_mask)
    laneProcess.combineImages(result, outimg, drawn_hotspots, drawn_lines_regions, warped_or,raw_lane)
    laneProcess.canDraw=False
    return result

writer = None
global frameArray
frameArray = []

while capture.isOpened():
    #ret, frame1 = capture.read()
    ret, frame = capture.read()
    if not ret:
        break

    # if frame is read correctly ret is True
    frame =cv.resize(frame,(1280,720))
    frame =SignDetector.DetectSign(frame)
    frame = Detect_lane(frame)
    frameArray.append(frame)
    cv.imshow('frame', frame)

    counter += 1

    # Stopping timer for FPS
    # Getting current time point in seconds
    fps_stop = timer()

    # Checking if timer reached 1 second
    if fps_stop - fps_start >= 1.0:
        # Showing FPS rate
        print('FPS rate is: ', counter)
        # Reset FPS counter
        counter = 0
        # Restart timer for FPS
        fps_start = timer()
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()

cv.destroyAllWindows()
print("writing video ")
fpsCount=25
def writeVideo():
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter('result_higwaytest.mp4', fourcc, fpsCount,
                            (1280, 720), True)
    print(len(frameArray))
    start=0
    for i in range(0,len(frameArray)):

        if((start+fpsCount)<len(frameArray)-1):
            end=start+fpsCount
            for j in tqdm(range(start,end)):
                writer.write(frameArray[j])

        start=start+fpsCount
    writer.release()
writeVideo()
print("done")