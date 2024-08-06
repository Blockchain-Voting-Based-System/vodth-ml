import cv2
import numpy as np
import imutils

def process_image(image):
    # Convert image from RGB to BGR and then to grayscale
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to a height of 600 pixels
    img = imutils.resize(img, height=600)

    # Apply Gaussian blur to the image
    gray = cv2.GaussianBlur(img, (3, 3), 0)

    # Apply a blackhat morphological operation to find dark regions on a light background
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    # Compute the gradient of the blackhat image
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)

    # Normalize the gradient image
    minVal, maxVal = np.min(gradX), np.max(gradX)
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # Apply a closing operation to the gradient image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    # Threshold the gradient image
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Apply another closing operation to the thresholded image
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    # Erode the thresholded image
    thresh = cv2.erode(thresh, None, iterations=3)

    # Remove noise from the borders of the image
    p = int(img.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, img.shape[1] - p:] = 0

    # Find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Iterate over the contours to find the region of interest (ROI)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        
        # Check if the aspect ratio of the contour is within a certain range
        if 5 < ar < 10:
            pX = int((x + w) * 0.05)
            pY = int((y + h) * 0.10)
            x, y = x - pX, y - pY
            w, h = w + (pX * 2), h + (pY * 2)
            
            # Extract the ROI from the image
            roi = img[y:y + h, x:x + w]
            break

    return roi

def deskew(cvImage):
    def getSkewAngle(cvImage) -> float:
        # Copy the image and apply Gaussian blur
        newImage = cvImage.copy()
        blur = cv2.GaussianBlur(newImage, (3, 3), 0)

        # Threshold the blurred image
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Dilate the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(thresh, kernel, iterations=3)

        # Find contours in the dilated image
        contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largestContour = contours[0]

        # Compute the minimum area rectangle for the largest contour
        minAreaRect = cv2.minAreaRect(largestContour)
        angle = minAreaRect[-1]

        # Adjust the angle
        if angle < -45:
            angle = 90 + angle
        if angle > 45:
            angle = 90 - angle
            angle = -1 * angle

        return -1.0 * angle

    def rotateImage(cvImage, angle: float):
        # Copy the image and compute its center
        newImage = cvImage.copy()
        h, w = newImage.shape[:2]
        center = (w // 2, h // 2)

        # Compute the rotation matrix and apply it to the image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return newImage

    # Get the skew angle and rotate the image to deskew it
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)
