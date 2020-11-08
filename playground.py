import cv2 as cv
import numpy as np
# read file
img = cv.imread("resources/matrix1.jpeg")


# define kernel
kernel = np.ones((5,5), np.uint8)

# color manipulation
imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# apply filer
imgBlur = cv.GaussianBlur(imgGrey, (9,9),0)
imgCanny = cv.Canny(imgGrey, 100, 100)

# apply filer with kernel (matrix)
imgDialation = cv.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv.erode(imgDialation, kernel, iterations=1)

# Resizing
print(img.shape)
imgRezised = cv.resize(img, (200,300))

# find contour
# contours,hierarchy = cv.findContours(imgGrey,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
width , height = 40, 60
pt1 = np.float32([[97,196],[143,185],[99,259],[152,241]])
pt2 = np.float32([[0,0],[width,0],[0,height],[width ,height]])
linearTransform = cv.getPerspectiveTransform( pt1, pt2)
imgTransform = cv.warpPerspective(imgGrey, linearTransform, (width, height))

# Stack images
imgStack = np.hstack((imgEroded, imgDialation))

# show
# cv.imshow("imgGrey", imgGrey)
# cv.imshow("imgBlur", imgBlur)
# cv.imshow("imgCanny", imgCanny)
# cv.imshow("imgDialation", imgDialation)
# cv.imshow("imgEroded", imgEroded)
# cv.imshow("imgResize", imgRezised)
cv.imshow("imgTransform", imgTransform)
cv.imshow("imgStack", imgStack)
cv.waitKey(0)
