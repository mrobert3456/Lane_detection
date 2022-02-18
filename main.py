import cv2 as cv
from timeit import default_timer as timer
from Lane import Lane
from Lane import LaneHistory
from Perspective_transform import Perspective
from Threshold import ImgThreshold


perspective =Perspective()
laneHistory =LaneHistory()
laneProcess = Lane(perspective)
Thresholder =ImgThreshold()

GoodLane = True
capture = cv.VideoCapture('ts_test.mp4')
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
    #binaryze the image
    combined_img = Thresholder.img_thres(image)

    #return  combined_img
    #get ROI
    roi_image = laneProcess.region_of_interest(combined_img)

    #return roi_image
    #Perspective transform
    warped = perspective.perspective_transform(roi_image, s_mask, dest_mask)
    #return blurred
    #return warped
    global GoodLane
    global  firstLane

    #if the lane is good then the marginsize =50, else marginsize=100
    if GoodLane:
        left_fit, right_fit, outimg,left_lane_inds, right_lane_inds = laneProcess.sliding_windown(warped,marginsize=50)  # returns the right and the left lane lines points
    else:
        left_fit, right_fit, outimg,left_lane_inds, right_lane_inds = laneProcess.sliding_windown(warped, marginsize=100)  # returns the right and the left lane lines points
    #return outimg


    drawn_lines_regions = laneProcess.draw_lane_lines_regions(warped)
    # cv.imshow('b',drawn_lines_regions)

    drawn_hotspots = laneProcess.draw_lines_hotspots(warped, left_lane_inds, right_lane_inds)
    # cv.imshow('c', drawn_hotspots)

    if laneProcess.sanity_check() or firstLane:
        result = laneProcess.draw_lines(warped, perspective=[s_mask, dest_mask], color=(0, 255, 0))
        GoodLane = True
        laneHistory.SetLeft_fit(left_fit)  # good lanes are added to the history
        laneHistory.SetRight_fit(right_fit)
        laneHistory.SetLeftCurvature(laneProcess.GetLeft_Curve())
        laneHistory.SetRightCurvature(laneProcess.GetRight_Curve())
        laneHistory.SetOffset(laneProcess.GetCenterOff())
        laneHistory.setWidth(laneProcess.getWidth())
        laneHistory.setRadius(laneProcess.getRadius())

        laneProcess.putDatasOnScreen(result)
        firstLane=False

    else:
        if laneHistory.getError()>12:
            laneHistory.setError(0)
            firstLane=True
            prev_l =laneHistory.GetLeft_fit()[-1]
            prev_r =laneHistory.GetRight_fit()[-1]

            laneHistory.GetLeft_fit().clear()
            laneHistory.GetRight_fit().clear()

            laneHistory.SetLeft_fit(prev_l)
            laneHistory.SetRight_fit(prev_r)

        result = laneProcess.draw_lines_fromHistory( warped, laneHistory.GetLeft_fit()[-1], laneHistory.GetRight_fit()[-1], perspective=[s_mask, dest_mask],color=(0, 255, 0))
        GoodLane = False
        laneHistory.incrementError()
        laneHistory.putHistoryDataOnScreen(result)

    roi_og = laneProcess.region_of_interest(image)
    warped_or = perspective.perspective_transform(roi_og, s_mask, dest_mask)
    laneProcess.combine_images(result, outimg, drawn_lines_regions, drawn_hotspots, warped_or)
    return result

while capture.isOpened():
    ret, frame = capture.read()
    # if frame is read correctly ret is True
    frame =cv.resize(frame,(1280,720))
    frame = Process_adv(frame)
    vmi = frame.shape[0]
    vmi3 = frame.shape[1]
    cv.imshow('frame', frame)
    #print("frame:"+str(fdb))
    #fdb+=1

    counter += 1

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