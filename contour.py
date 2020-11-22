import cv2 as cv

def find_contour(img, img2):
    """
    Take in a (Canny) image, find conture and return a new image with contour and bounding
    boxes around digit
    :param img: Canny Image
    :param img2: Original Image
    :return: Original Image with Bounding Box
    """
    imgCopy = img2.copy()
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    crop_anchors = []
    for contour in contours:
        area = cv.contourArea(contour)
        # give threshold of area to avoid noise
        if area > 20:
            # cv.drawContours(imgCopy, contour, -1 , (255, 0, 0), 1)
            peri = cv.arcLength(contour, False)
            appro = cv.approxPolyDP(contour, 0.0002*peri, True)
            x,y,width, height = cv.boundingRect(appro)
            # draw bounding box
            cv.rectangle(imgCopy, (x-5,y-5), (x+width+5, y+height+5), (0,255,0), thickness=1)
            crop_anchors.append(((y-5,y+height+5),(x-5,x+width+5)))
    return imgCopy, crop_anchors
