
# import the necessary packages
import numpy as np
import argparse
import cv2


def intAverage(lst):
    return int(sum(lst) / len(lst))

# load the image, clone it for output, and then convert it to grayscale
img = cv2.imread('data_images/database/clock.png')
image = cv2.resize(img,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)

output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,thresh=127,maxval=255,type=cv2.THRESH_BINARY_INV)
edges = cv2.Canny(thresh,100,200)
edges_output = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)

#     # show the output image
# cv2.imshow("output", np.hstack([thresh, edges]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# detect circles in the image
# https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 5,
              param1=300,
              param2=60,
              minRadius = 140,
              maxRadius=200)
# https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
lines = cv2.HoughLines(edges,1,np.pi/360,80)

for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(edges_output,(x1,y1),(x2,y2),(0,0,255),2)



# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.uint16(np.around(circles))
    print(circles)
    radius = []
    x0 = []
    y0 = []
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles[0,:]:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        radius.append(r)
        x0.append(x)
        y0.append(y)
        cv2.circle(edges_output, (x, y), r, (0, 255, 0), 1)
        # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.circle(edges_output, (intAverage(x0), intAverage(y0)), intAverage(radius), (0, 0, 255), 2)
    cv2.rectangle(edges_output, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 255), -1)

    # show the output image
    cv2.imshow("output", np.hstack([image, edges_output]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()