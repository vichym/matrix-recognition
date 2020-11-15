import cv2 as cv

# read file
img = cv.imread("resources/matrix1.jpeg")
imgCopy = img.copy()

## preprocessing
# convert to grey scale
imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Blurr the image
imgBlur = cv.GaussianBlur(imgGrey, (7,7),1)
# Blurr the image
imgBlur = cv.GaussianBlur(imgGrey, (7,7),1)

# Canny image
imgCanny = cv.Canny(imgBlur, 50, 50 )


# Get contour
def find_contour(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        # give threshold of area to avoid noise
        if area > 20:
            cv.drawContours(imgCopy, contour, -1 , (255, 0, 0), 3)
            peri = cv.arcLength(contour, True)
            appro = cv.approxPolyDP(contour, 0.02*peri, True)
            x,y,width, height = cv.boundingRect(appro)
            # draw bounding box
            cv.rectangle(imgCopy, (x,y), (x+width, y+height), (0,255,0), thickness=1)

# find contours and draw bounding boxes
find_contour(imgCanny)

cv.imshow("img", img)
cv.imshow("imgGrey", imgGrey)
cv.imshow("imgBlur", imgBlur)
# cv.imshow("imgCanny", imgCanny)
cv.imshow("imgContour", imgCopy)

cv.waitKey(15000)
