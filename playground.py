import cv2 as cv
import numpy as np
from pixelate import pixelate
from contour import find_contour
# read file
img = cv.imread("resources/matrix1.jpeg")

# define kernel
kernel = np.ones((15,15), np.uint8)

# color manipulation
imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# apply filer
imgBlur = cv.GaussianBlur(imgGrey, (9,9),10)
imgCanny = cv.Canny(imgGrey, 100, 100)

# apply filer with kernel (matrix)
# imgDialation = cv.dilate(imgBlur, kernel, iterations=1)
imgEroded = cv.erode(imgBlur, kernel, iterations=2)
imgEroded = cv.Canny(imgEroded, 100, 100)
imgContour = find_contour(imgEroded,img)

# # Resizing
# print(img.shape)
# imgRezised = cv.resize(img, (200,300))

# find contour
# contours,hierarchy = cv.findContours(imgGrey,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# width , height = 40, 60
# pt1 = np.float32([[97,196],[143,185],[99,259],[152,241]])
# pt2 = np.float32([[0,0],[width,0],[0,height],[width ,height]])
# linearTransform = cv.getPerspectiveTransform( pt1, pt2)
# imgTransform = cv.warpPerspective(imgGrey, linearTransform, (width, height))

# Stack images
# imgStack = np.hstack((imgEroded, imgDialation))


# pixelate
# pixelate(imgGrey, 32, 32)
# pixelate(imgBlur, 32, 32)
# pixelate(imgCanny, 32, 32)
# pixelate(imgDialation, 32, 32)
# pixelate(imgEroded, 32, 32)


# show
cv.imshow("imgGrey", imgGrey)
cv.imshow("imgBlur", imgBlur)
# cv.imshow("imgCanny", imgCanny)
# cv.imshow("imgDialation", imgDialation)
cv.imshow("imgEroded", imgEroded)
cv.imshow("imgContour", imgContour)
# //
# cv.imshow("imgStack", imgStack)
cv.waitKey(0)
