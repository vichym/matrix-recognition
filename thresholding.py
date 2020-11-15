import cv2
import numpy as np
from pixelate import pixelate

img = cv2.imread('resources/matrix1.jpeg')
cv2.imshow("Original", img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grey", img)

ret,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
cv2.imshow("Threshold", img)

img = cv2.GaussianBlur(img, (3,3),1)
cv2.imshow("Blurr1", img)

img = pixelate(img, 12, 12)
cv2.imshow("Pixelate 1", img)

img = cv2.GaussianBlur(img, (25,25),10)
cv2.imshow("Blurr2", img)

# img = pixelate(img, 15, 15)
# cv2.imshow("Pixelate 2", img)

# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,1)
# cv2.imshow("Adaptive Threshold", img)

cv2.imshow("Pixel", img)

cv2.waitKey(50000)