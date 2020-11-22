import cv2 as cv
import numpy as np
from contour import *

def run(image, K , D):
    img = cv.imread(image)
    imgCopy = img.copy()

    h,w,channel = img.shape

    # Kernal, Blur and Dialation coefficients
    # K = 0.0008
    # D = 0.0005
    # convert to grey scale
    imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Blurr the image
    imgBlur = cv.GaussianBlur(imgGrey, (51,51),1)

    ret,imgThreshold = cv.threshold(imgBlur,100,255,cv.THRESH_BINARY)

    # Kernel for eroding and dilating
    kernel = np.ones((int(h*K),int(w*K)), np.uint8)

    # Erode: expand the black pixels and blend sign and number together in the image
    imgEroded = cv.erode(imgThreshold, kernel, iterations=2)

    # Dilalate: narrow the black blob back
    imgDialation = cv.dilate(imgEroded, kernel, iterations=2)

    # Canny: get the outter boundary of the group
    imgCanny = cv.Canny(imgDialation, D*(h+w), D*(h+w) )

    # Contour: get the contour and paint bounding boxes around
    imgFinal, crop_points = find_contour(imgCanny, imgCopy)

    for points in crop_points:
        c = img[points[0][0]:points[0][1], points[1][0]:points[1][1]]
        # cv.imshow("{}".format(c), c)
        print(points[0][0],points[0][1])

    # cv.imshow("img", img)
    # cv.imshow("imgGrey", imgGrey)
    # cv.imshow("imgBlur", imgBlur)
    # cv.imshow("imgThreshold", imgThreshold)
    cv.imshow("imgErode", imgEroded)
    # cv.imshow("imgDila", imgDialation)
    cv.imshow("imgCanny", imgCanny)
    # cv.imshow("imgContour", imgFinal)

    cv.waitKey(0)
if __name__ == '__main__':
    for K in [0.01,0.001,0.0001,0.00001]:
        for D in [0.01,0.001,0.0001,0.00005]:
            run("./resources/matrix4.png", K,D)
