
from TrafficSign_recognition import TrafficSignDetector
import os
import pandas as pd
import cv2 as cv

SignDetector = TrafficSignDetector()

def getTestImages():
    """
    read the video input and saves every second frame in the 'teszt' folder
    Need to filter out images manually in which zero signs can be found!!!
    """
    capture = cv.VideoCapture('ts_test2.mp4')
    frameCount=0
    while capture.isOpened():
        ret1, frame1 = capture.read()
        ret, frame = capture.read()
        if not ret:
            break
        cv.imwrite('teszt/'+'tesztKep'+str(frameCount)+'.jpg',frame)
        frameCount+=1

def testTSD_TSR(imagesPath,savePath):
    """
    iterates through the 'test' folder and detects the traffic sign in them and splits the resulted frames
    into 'Valid' and 'Invalid' folder according to the test results
    """
    os.chdir(imagesPath)
    detectionCounter = 0
    invalidDetection=0
    processedImgs =[]
    processedDetCount =[]
    for current_dir, dirs, files in os.walk('.'):
        # Going through all files
        for f in files:
            if f.endswith('.jpg') and f not in processedImgs:
                try:
                   frame = cv.imread(f)
                   frame, results = SignDetector.recognizeTrafficSign(frame)
                   processedImgs.append(f)
                   #if detections are made, then recognise the traffic signs
                   if len(results)>0:
                       detectionCounter += len(results.flatten()) # number of detected traffic signs
                       fname = savePath+'Valid/'+str(len(results))+'_db_' + str(f) # every image name will contain how many traffic signes had been detected
                       cv.imwrite(fname, frame)

                       processedDetCount.append(len(results))

                   else:
                       invalidDetection+=1
                       fname = savePath+'Invalid/' + str(f)
                       cv.imwrite(fname, frame)
                       processedDetCount.append(0)
                except:
                    print('')

    #writes the valid detection frames and its detection count to a txt, to convert them into csv file
    with open('testhelper.txt', 'w') as f:
       for i in range(0,len(processedImgs)):
           f.write(processedImgs[i]+';'+str(processedDetCount[i])+'\n')

    tesztdf = pd.read_csv('testhelper.txt',names=['Image','detectionCount'],sep=';' )
    tesztdf.to_csv('testResults.csv')
    print('invalid detections: '+ str(invalidDetection))
    print('detection count: '+str(detectionCounter))


testTSD_TSR('tesztTSR/','processed/')
