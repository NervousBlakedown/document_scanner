# Document Scanner.
# Imports
import numpy as np
import argparse
import cv2
from skimage.filters import threshold_local
import imutils

def order_points(pts):
    # initialzie a list of coordinates that will be ordered such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def edge_detection(image_path, height=500, blur_kernel=(5,5), show_steps=False):
    # Load the image, clone it for output, and then resize it
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not open or find the image. Please check the path.")
    
    ratio = image.shape[0] / height
    orig = image.copy()
    image = imutils.resize(image, height=height)

    # Convert to grayscale, blur, and edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_kernel, 0)
    edged = cv2.Canny(gray, 75, 200)

    if show_steps:
        cv2.imshow("Image", image)
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return edged, orig, ratio

def find_contours(edged, orig, ratio, show_steps=False):
    # Find contours and keep the largest ones
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    # Loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, we can assume we've found the paper
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        raise ValueError("Could not find the outline of the paper. Please check the image or try another one.")

    if show_steps:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return screenCnt, ratio

def transform_and_threshold(orig, screenCnt, ratio):
    # Apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # Convert the warped image to grayscale, then threshold it to give it the 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255

    return warped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    ap.add_argument("-s", "--show", type=bool, default=False, help="Show steps")
    args = vars(ap.parse_args())

    # Execute edge detection
    edged, orig, ratio = edge_detection(args["image"], show_steps=args["show"])

    # Find contours and get the screen contour of the document
    screenCnt, ratio = find_contours(edged, orig, ratio, show_steps=args["show"])

    # Apply transformation and thresholding to get the scanned effect
    scanned = transform_and_threshold(orig, screenCnt, ratio)

    # Show the original and the scanned images
    cv2.imshow("Original", imutils.resize(orig, height=650))
    cv2.imshow("Scanned", imutils.resize(scanned, height=650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

