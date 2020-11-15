import cv2 as cv # Input image

def pixelate(input, w, h):
    # Get input size
    height, width = input.shape[:2]

    # Resize input to "pixelated" size
    temp = cv.resize(input, (w, h), interpolation=cv.INTER_NEAREST)
    temp2 = cv.resize(temp, (w, h), interpolation=cv.INTER_MAX)

    # Initialize output image
    output = cv.resize(temp2, (width, height), interpolation=cv.INTER_NEAREST)

    return output