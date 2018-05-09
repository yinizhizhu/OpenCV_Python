import numpy as np
import cv2
from matplotlib import pyplot as plt


step = 60
delta = 30


def showImg(name, img, x):
    print x
    cv2.namedWindow(name)
    cv2.moveWindow(name, x=x, y=x)
    cv2.imshow(name, img)

show1 = 0
img = cv2.imread('pic/ArtRoom.png')
if show1:
    showImg('img', img, step)
    step += delta

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
if show1:
    showImg("gray", gray, step)
    step += delta

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
if show1:
    print ret
    print thresh
    showImg("thresh", thresh, step)
    step += delta
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show2 = 0
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
if show2:
    showImg('opening', opening, step)
    step += delta

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
if show2:
    showImg('sure_bg', sure_bg, step)
    step += delta

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
if show2:
    showImg('sure_fg', sure_fg, step)
    step += delta

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
if show2:
    showImg('unknown', unknown, step)
    step += delta

if show2:
    cv2.waitKey(0)
    cv2.destroyAllWindows()

show3 = 1
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
if show3:
    print ret
    print markers
    showImg('markers', markers, step)
    step += delta
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
if show3:
    print ret
    print markers
    showImg('markers', markers, step)
    step += delta
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
if show3:
    print ret
    print markers
    showImg('markers', markers, step)
    step += delta
    cv2.waitKey(0)
    cv2.destroyAllWindows()

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
if show3:
    print ret
    print markers
    showImg('markers', markers, step)
    step += delta
    cv2.waitKey(0)
    cv2.destroyAllWindows()