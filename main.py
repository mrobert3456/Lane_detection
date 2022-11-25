
import cv2 as cv
from timeit import default_timer as timer
from Lane import Lane
from Lane import LaneHistory
from Perspective_transform import Perspective
from Threshold import ImgThreshold
from TrafficSign_recognition import TrafficSignDetector
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

perspective =Perspective()
laneHistory =LaneHistory()
laneProcess = Lane(perspective)
Thresholder =ImgThreshold()
SignDetector = TrafficSignDetector()
GoodLane = True
firstLane=True




def Detect_lane(image, useHistory):

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

    global usedHistory
    usedHistory=False
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
    result=image
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

    elif(useHistory):
        GoodLane = False
        # if detection error count is above 20
        if laneHistory.getError()>20:
            laneHistory.setError(0)
            firstLane=True
            #laneProcess.canDraw = False
            laneHistory.getLeftFit().clear()
            laneHistory.getRightFit().clear()

        #if history has data
        if len(laneHistory.getLeftFit()) >0 and len(laneHistory.getRightFit())>0:
            result = laneProcess.drawLinesFromHistory( warped,  laneHistory.getLeftFit()[-1], laneHistory.getRightFit()[-1],laneHistory.ploty, perspective=[s_mask, dest_mask],color=(0, 255, 255))
            laneHistory.incrementError()
            result  = laneProcess.displayHeadingLine(result,laneHistory.getDirection())
            laneHistory.putHistoryDataOnScreen(result)
            usedHistory = True
        else:
            useHistory=False
            result = image
    roi_og = laneProcess.regionOfInterest(image)
    warped_or = perspective.perspective_transform(roi_og, s_mask, dest_mask)
    laneProcess.combineImages(result, outimg, drawn_hotspots, drawn_lines_regions, warped_or,raw_lane)
    #laneProcess.canDraw=False
    return result,validLane, usedHistory

writer = None
global frameArray
frameArray = []

def processVideo():
    # FPS counter
    counter = 0
    fps_start = timer()
    capture = cv.VideoCapture('higwaytest.mp4')
    laneProcess.setUseKalman(True)
    while capture.isOpened() :
        ret1, frame1 = capture.read()
        ret, frame = capture.read()
        if not ret:
            break

        # if frame is read correctly ret is True
        frame =cv.resize(frame,(1280,720))
        #frame =SignDetector.recognizeTrafficSign(frame)
        frame,validLane, usedHist = Detect_lane(frame,False)
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

def writeVideo():
    fpsCount = 25
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

def getTestImages():
    for i in range(0, len(frameArray)):
        cv.imwrite('teszt/'+'tesztKep'+str(i)+'.jpg',frameArray[i])

def testTSDetection():
    os.chdir('teszt/')
    detectionCounter = 0
    invalidDetection=0
    validImg =[]
    validImgDc =[]
    for current_dir, dirs, files in os.walk('.'):
        # Going through all files
        for f in files:
            if f.endswith('.jpg'):
                try:
                   frame = cv.imread(f)
                   frame, results = SignDetector.recognizeTrafficSign(frame)

                   if len(results)>0:
                       detectionCounter += len(results.flatten())
                       fname = 'feldolgozas/Valid/'+str(len(results))+'_db_' + str(f)
                       cv.imwrite(fname, frame)
                       validImg.append(f)
                       validImgDc.append(len(results))

                   else:
                       invalidDetection+=1
                       fname = 'feldolgozas/Invalid/' + str(f)
                       cv.imwrite(fname, frame)
                       print(f)
                except:
                    print('no img left')

    with open('tesztered.txt', 'w') as f:
       for i in range(0,len(validImg)):
           f.write(validImg[i]+';'+str(validImgDc[i])+'\n')

    tesztdf = pd.read_csv('tesztered.txt',names=['Image','detectionCount'],sep=';' )
    tesztdf.to_csv('teszResults.csv')
    print('invalid detections: '+ str(invalidDetection))
    print('detection count: '+str(detectionCounter))

def testLaneDetection(useKalman=True, useHistory = True):
        laneProcess.setUseKalman(useKalman)

        capture = cv.VideoCapture('ts_test2.mp4')
        frameCount = 0
        validCount=0
        invalidCount=0
        validImg=[]
        while capture.isOpened() and frameCount<1374:
            ret1, frame1 = capture.read()
            ret, frame = capture.read()
            if not ret:
                break
            #cv.imwrite('tesztLane/use_Kalman_History/' + 'tesztKep' + str(frameCount) + '.jpg', frame)
            frame,validLane,usedHist = Detect_lane(frame,useHistory)

            if validLane or usedHist:
                cv.imwrite('tesztLane/city_road/Valid/' + 'tesztKep' + str(frameCount) + '.jpg', frame)
                validCount+=1
                validImg.append(1)
            else:
                cv.imwrite('tesztLane/city_road/Invalid/' + 'tesztKep' + str(frameCount) + '.jpg', frame)
                invalidCount+=1
                validImg.append(0)

            frameCount+=1
            print('processed: '+str(frameCount)+'.frame')
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()

        with open('tesztLane/city_road/tesztered.txt', 'w') as f:
            for i in range(0, len(validImg)):
                f.write('tesztKep'+str(i) + ';' + str(validImg[i]) + '\n')

        tesztdf = pd.read_csv('tesztLane/city_road/tesztered.txt', names=['Image', 'isValid'], sep=';')
        tesztdf.to_csv('tesztLane/city_road/teszResults.csv')
        print('Valid lane detections: '+str(validCount))
        print('Invalid lane detections: '+str(invalidCount))

#processVideo()
#writeVideo()
#getTestImages()
#testTSDetection()
testLaneDetection()
print("done")