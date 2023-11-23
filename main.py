import sys

import cv2 as cv
from timeit import default_timer as timer
from TrafficSign_recognition import TrafficSignDetector
from LaneDetection import LaneDetection
from tqdm import tqdm



writer = None
global frameArray
frameArray = []


def processVideo(video, includeTsr=False):
    # FPS counter
    counter = 0
    fps_start = timer()
    capture = cv.VideoCapture(video)

    while capture.isOpened():
        #ret1, frame1 = capture.read()
        ret, frame = capture.read()
        if not ret:
            break

        # if frame is read correctly ret is True
        frame = cv.resize(frame, (1280, 720))
        if includeTsr:
            frame, results = SignDetector.recognizeTrafficSign(frame)
        frame, validLane, usedHist = LaneDetector.detectLane(frame)
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


def writeVideo(resVideoName):
    fpsCount = 25
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter(resVideoName, fourcc, fpsCount,
                            (1280, 720), True)
    print(len(frameArray))
    start = 0
    for i in range(0, len(frameArray)):

        if ((start + fpsCount) < len(frameArray) - 1):
            end = start + fpsCount
            for j in tqdm(range(start, end)):
                writer.write(frameArray[j])

        start = start + fpsCount
    writer.release()


if __name__ == '__main__':

    LaneDetector = LaneDetection(useKalman=True, useHistory=True)
    argCount = len(sys.argv)

    inputFile = str(sys.argv[1]) if argCount >= 2 else 'higwaytest.mp4'
    includeTsr = str(sys.argv[2]) if argCount >= 3 else 'lane' #combined
    outputFile = str(sys.argv[3]) if argCount >= 4 else 'output.mp4'

    if includeTsr=='combined':
        SignDetector = TrafficSignDetector()
        processVideo(inputFile, True)

    else:
        processVideo(inputFile)

    writeVideo(outputFile)
    print("done")
