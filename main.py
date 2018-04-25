import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('pic/ArtRoom.png', 1)          # queryImage
print img1.shape

img2 = cv2.imread('pic/ArtRoom_0.75_scl.png', 1)          # queryImage
print img2.shape

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
# keypoints sorted by the x
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
print len(kp1)
# for kp in kp1:
#     print kp.pt

print len(kp2)

# BFMatcher with default params
bf = cv2.BFMatcher()

# match(queryImage, trainImage)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

for m in matches:
    print m.distance, m.imgIdx, m.queryIdx, m.trainIdx

# cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
#
# plt.figure()
# plt.imshow(img3)
# plt.show()