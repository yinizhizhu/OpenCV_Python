import sys
import cv2

basis = ['pic/ArtRoom', '_0.75_cr', '_0.75_sv', '_0.75_multiop',
         '_0.75_sc', '_0.75_scl', '_0.75_sm', '_0.75_sns', '_0.75_warp']

names = []
names.append(basis[0] + '.png')
for i in xrange(1, len(basis)):
    names.append(basis[0]+basis[i]+'.png')

print names

# speed-up using multithreads
# cv2.setUseOptimized(True)
# cv2.setNumThreads(4)

for name in names:
    # read image
    im = cv2.imread(name)

    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # set input image on which we will run segmentation
    ss.setBaseImage(im)

    # Switch to fast but low recall Selective Search method
    # ss.switchToSelectiveSearchFast()

    # Switch to high recall but slow Selective Search method
    ss.switchToSelectiveSearchQuality()


    # run selective search segmentation on input image
    rects = ss.process()

    print name
    print('    Total Number of Region Proposals: {}'.format(len(rects)))

    # number of region proposals to show
    numShowRects = 1
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 1

    while True:
        # create a copy of original image
        imOut = im.copy()

        # itereate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # show output
        cv2.imshow("Output", imOut)

        # record key press
        k = cv2.waitKey(0) & 0xFF

        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()
