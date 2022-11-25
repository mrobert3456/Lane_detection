from Lane import Lane
from Lane import LaneHistory
from Perspective_transform import Perspective
from Threshold import ImgThreshold


class LaneDetection:
    def __init__(self, useKalman, useHistory):
        self.perspective = Perspective()
        self.laneHistory = LaneHistory()
        self.laneProcess = Lane(self.perspective)
        self.Thresholder = ImgThreshold()
        self.GoodLane = True
        self.firstLane = True
        self.useHistory = useHistory
        self.laneProcess.setUseKalman(useKalman)

    def detectLane(self, image):
        self.laneProcess.SetImg(image)
        self.perspective.setImg(image)

        dest_mask = self.perspective.setDestination()
        s_mask = self.perspective.setSource()

        # get ROI
        roi_image = self.laneProcess.regionOfInterest(image)

        # binaryze the image
        combined_img = self.Thresholder.img_thres(roi_image)

        # Perspective transform
        warped = self.perspective.perspective_transform(combined_img, s_mask, dest_mask)

        global usedHistory  # determin whether history data was used
        usedHistory = False

        # if the lane is good then the marginsize =50, else marginsize=100
        if self.GoodLane:
            left_fit, right_fit, outimg, left_lane_inds, right_lane_inds = self.laneProcess.slidingWindown(warped,
                                                                                                           marginsize=25)  # returns the right and the left lane lines points
        else:
            left_fit, right_fit, outimg, left_lane_inds, right_lane_inds = self.laneProcess.slidingWindown(warped,
                                                                                                           marginsize=50)  # returns the right and the left lane lines points

        # draw detection steps results to the input frame
        drawn_lines_regions = self.laneProcess.drawLaneLinesRegions(warped)

        drawn_hotspots = self.laneProcess.drawLinesHotspots(warped, left_lane_inds, right_lane_inds)

        raw_lane = self.laneProcess.drawLines(warped, perspective=[s_mask, dest_mask], color=(0, 255, 0))

        # sanity check
        result = image
        validLane = self.laneProcess.sanityCheck()
        if validLane or self.firstLane:
            result = raw_lane
            if validLane:  # good lanes are added to the history
                self.GoodLane = True
                self.laneHistory.setHistory(0, left_fit, right_fit, self.laneProcess.getLeftCurve(),
                                            self.laneProcess.getRightCurve(),
                                            self.laneProcess.getCenterOff(), self.laneProcess.getWidth(),
                                            self.laneProcess.getRadius(),
                                            self.laneProcess.getPloty(), self.laneProcess.getDirection())
                self.laneProcess.putDatasOnScreen(result)
                result = self.laneProcess.displayHeadingLine(result, self.laneProcess.getDirection())
            self.firstLane = False

        elif (self.useHistory):
            self.GoodLane = False
            # if detection error count is above 20
            if self.laneHistory.getError() > 20:
                self.laneHistory.setError(0)
                self.firstLane = True
                # laneProcess.canDraw = False
                self.laneHistory.getLeftFit().clear()
                self.laneHistory.getRightFit().clear()

            # if history has data
            if len(self.laneHistory.getLeftFit()) > 0 and len(self.laneHistory.getRightFit()) > 0:
                result = self.laneProcess.drawLinesFromHistory(warped, self.laneHistory.getLeftFit()[-1],
                                                               self.laneHistory.getRightFit()[-1],
                                                               self.laneHistory.ploty,
                                                               perspective=[s_mask, dest_mask], color=(0, 255, 255))
                self.laneHistory.incrementError()
                result = self.laneProcess.displayHeadingLine(result, self.laneHistory.getDirection())
                self.laneHistory.putHistoryDataOnScreen(result)
                usedHistory = True
            else:
                usedHistory = False
                result = image
        else:
            usedHistory = False
        roi_og = self.laneProcess.regionOfInterest(image)
        warped_or = self.perspective.perspective_transform(roi_og, s_mask, dest_mask)
        self.laneProcess.combineImages(result, outimg, drawn_hotspots, drawn_lines_regions, warped_or, raw_lane)
        # laneProcess.canDraw=False
        return result, validLane, usedHistory
