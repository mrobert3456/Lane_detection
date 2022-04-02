import numpy as np
import cv2 as cv

class Perspective:
    def __init__(self):
        self.image =[]
        self.destination =[]
        self.source =[]

    def getDestination(self):
        return self.destination

    def getSource(self):
        return self.source

    def getImg(self):
        return self.image

    def setImg(self,img):
        self.image=img
        return

    def setDestination(self):
        """Returns a matrix with the quadrangle indicies in the destination image """
        # _imageSize = (1280, 720)
        xOffset = self.image.shape[1] / 4
        yOffset = 70
        destinationBottomLeft = (xOffset, self.image.shape[0])
        destinationBottomRight = (self.image.shape[1] - xOffset, self.image.shape[0])
        destinationTopLeft = (xOffset, yOffset)
        destinationTopRight = (self.image.shape[1] - xOffset, yOffset)
        return np.float32([destinationBottomLeft, destinationTopLeft, destinationTopRight, destinationBottomRight])

    def setSource(self):
        """Returns a matrix with the quadrangle indicies from the original image """
        # _imageSize = (1280, 720)
        #self.image.shape = 720,1280,3
        xOffsetBottom = 100 #200
        xOffsetMiddle =475 #595
        yOffset = 375#450
        sourceBottomLeft = (xOffsetBottom, self.image.shape[0])
        sourceBottomRight = (self.image.shape[1] - xOffsetBottom, self.image.shape[0])
        sourceTopLeft = (xOffsetMiddle, yOffset)
        sourceTopRight = (self.image.shape[1] - xOffsetMiddle, yOffset)

        return np.float32([sourceBottomLeft, sourceTopLeft, sourceTopRight, sourceBottomRight])


    def perspective_transform(self,img,src_m, dest_m):
        """Gets the bird eye view of the image"""
        """From a bird eye view, the lane lines can be seen as parallel, as it is, but from the original view it seems as the lines are coming together in distance"""
        img_size = (img.shape[1], img.shape[0])
        #src = np.float32(src_m)  # source transformation matrix
        #dest = np.float32(dest_m)  # destination transformation matrix
        M = cv.getPerspectiveTransform(src_m, dest_m)  # gets the perpective transformation matrix
        warped_img = cv.warpPerspective(img, M, img_size)  # warps the image #, flags=cv.INTER_LINEAR
        return warped_img