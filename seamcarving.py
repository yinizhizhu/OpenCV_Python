# import the necessary packages
from skimage import transform
from skimage import filters
import cv2

# load the image and convert it to grayscale
image = cv2.imread('pic/ArtRoom.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# compute the Sobel gradient magnitude representation
# of the image -- this will serve as our "energy map"
# input to the seam carving algorithm
mag = filters.sobel(gray.astype("float"))

# show the original image
cv2.imshow("Original", image)

# loop over a number of seams to remove
for numSeams in range(20, 140, 20):
    # perform seam carving, removing the desired number
    # of frames from the image -- `vertical` cuts will
    # change the image width while `horizontal` cuts will
    # change the image height
    carved = transform.seam_carve(image, mag, 'vertical',
                                  numSeams)
    print("[INFO] removing {} seams; new size: "
          "w={}, h={}".format(numSeams, carved.shape[1],
                              carved.shape[0]))

    # show the output of the seam carving algorithm
    cv2.imshow("Carved", carved)
    cv2.waitKey(0)