import numpy as np
import cv2

# Create a black image
img = np.zeros((512,512,3), np.uint8)
#
# # Draw a diagonal blue line with thickness of 5 px
# cv2.line(img,(0,0),(511,511),(255,0,0),5)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.rectangle(img,(384,0),(510,128),(0,255,0),1)

cv2.rectangle(img,(0,90),(180,180),(255,0,0),1)

cv2.rectangle(img,(99,0),(189,90),(0,0,255),1)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#
# cv2.circle(img,(447,63), 63, (0,0,255), -1)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
# cv2.polylines(img,[pts],True,(0,255,255))
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,'OpenCV',(10,500), font, 0.1,(255,255,255),2,cv2.LINE_AA)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
