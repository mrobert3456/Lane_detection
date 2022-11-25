
import cv2 as cv
import pandas as pd
from LaneDetection import LaneDetection

LaneDetector = LaneDetection(useKalman=True, useHistory=True)


def testLaneDetection( video, savePath, maxFrameCount,useKalman=True, useHistory=True, higwayTest=True):
    """
    reads a video input and detects the Lane
    splits the results into 'Valid' and 'Invalid' folders according to the results
    """
    subName=''
    if higwayTest:
        subName='higway_road/'
    else:
        subName='city_road/'

    if useKalman and useHistory:
        subName+='use_Kalman_History/'
    elif useKalman:
        subName+='use_Kalman/'
    elif useHistory:
        subName+='use_History/'


    capture = cv.VideoCapture(video)
    frameCount = 0
    validCount = 0
    invalidCount = 0
    validImg = []
    while capture.isOpened() and frameCount < maxFrameCount:
        ret1, frame1 = capture.read()
        ret, frame = capture.read()
        if not ret:
            break
        frame, validLane, usedHist = LaneDetector.detectLane(frame)

        if validLane or usedHist: # if the detection is valid or History data made it valid
            cv.imwrite(savePath+subName+'Valid/' + 'tesztKep' + str(frameCount) + '.jpg', frame)
            validCount += 1
            validImg.append(1)
        else:
            cv.imwrite(savePath+subName+'Invalid/' + 'tesztKep' + str(frameCount) + '.jpg', frame)
            invalidCount += 1
            validImg.append(0)

        frameCount += 1
        print('processed: ' + str(frameCount) + '.frame')
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()

    with open(savePath+subName+'testhelper.txt', 'w') as f:
        for i in range(0, len(validImg)):
            f.write('tesztKep' + str(i) + ';' + str(validImg[i]) + '\n')

    tesztdf = pd.read_csv(savePath+subName+'testhelper.txt', names=['Image', 'isValid'], sep=';')
    tesztdf.to_csv(savePath+subName+'testResults.csv')
    print('Valid lane detections: ' + str(validCount))
    print('Invalid lane detections: ' + str(invalidCount))



testLaneDetection('ts_test2.mp4','tesztLane/',1374, higwayTest=False)

testLaneDetection('higwaytest.mp4','tesztLane/',900,useKalman=False, higwayTest=True)
testLaneDetection('higwaytest.mp4','tesztLane/',900,useHistory=False, higwayTest=True)
testLaneDetection('higwaytest.mp4','tesztLane/',900,higwayTest=True)

