import cv2, random
import scipy.io as scio
import numpy as np
from matplotlib import pyplot as plt


def readImg(filename, h1, h2, w1, w2):
    """"
    Color image loaded by OpenCV is in BGR mode, but Matplotlib displays in RGB mode.
    cv2.imread(path, style)
        1 - cv2.IMREAD_COLOR
        0 - cv2.IMREAD_GRAYSCALE
        -1 - cv2.IMREAD_UNCHANGED
    """
    img = cv2.imread(filename, 1)
    # plt.figure()
    # plt.imshow(img)
    img = img[h1:h2, w1:w2]
    return img


def readMat(filename, h1, h2, w1, w2):
    data = scio.loadmat(filename)
    data = data['mapstage2']
    print data
    print data.shape, '-', data.dtype, data.size
    return data[h1:h2, w1:w2]


def siftMatch(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    # keypoints sorted by the x
    kpts1, des1 = sift.detectAndCompute(img1, None)
    kpts2, des2 = sift.detectAndCompute(img2, None)
    print len(kpts1)
    # for kp in kp1:
    #     print kp.pt

    print len(kpts2)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    # match(queryImage, trainImage)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatches(img1, kpts1, img2, kpts2, matches, None, flags=2)
    plt.figure()
    plt.imshow(img3)

    """
    :param matches: remove the mismatch pairs
    :return:
    """
    # tolerance = 1 # used when comparing similarity matrices of match pairs
    # #  best_match = 0
    # consensus_set = []
    # best_sim = []
    # # find consensus of translation between a random keypoint and the rest for
    # # a number of times to find the best match regarding translation
    # for i in range(100):
    #     idxs = random.sample(range(len(matches)), 2)
    #     # calc similarity between kp1 and kp2
    #     kp11 = kpts1[matches[idxs[0]].queryIdx]
    #     kp12 = kpts2[matches[idxs[0]].trainIdx]
    #     kp21 = kpts1[matches[idxs[1]].queryIdx]
    #     kp22 = kpts2[matches[idxs[1]].trainIdx]
    #     A = np.array([[kp11.pt[0], -kp11.pt[1], 1, 0],
    #                   [kp11.pt[1], kp11.pt[0], 0, 1],
    #                   [kp21.pt[0], -kp21.pt[1], 1, 0],
    #                   [kp21.pt[1], kp21.pt[0], 0, 1]])
    #     b = np.array([kp12.pt[0], kp12.pt[1], kp22.pt[0], kp22.pt[1]])
    #     Sim,_,_,_ = np.linalg.lstsq(A, b)
    #     Sim = np.array([[Sim[0], -Sim[1], Sim[2]],
    #                     [Sim[1], Sim[0], Sim[3]]])
    #     temp_consensus_set = []
    #     for j in range(len(matches) - 1):
    #         match = matches[j]
    #         kp11 = kpts1[matches[j].queryIdx]
    #         kp12 = kpts2[matches[j].trainIdx]
    #         kp21 = kpts1[matches[j+1].queryIdx]
    #         kp22 = kpts2[matches[j+1].trainIdx]
    #         A = np.array([[kp11.pt[0], -kp11.pt[1], 1, 0],
    #                       [kp11.pt[1], kp11.pt[0], 0, 1],
    #                       [kp21.pt[0], -kp21.pt[1], 1, 0],
    #                       [kp21.pt[1], kp21.pt[0], 0, 1]])
    #         b = np.array([kp12.pt[0], kp12.pt[1], kp22.pt[0], kp22.pt[1]])
    #         Sim2,_,_,_ = np.linalg.lstsq(A, b)
    #         Sim2 = np.array([[Sim2[0], -Sim2[1], Sim2[2]],
    #                          [Sim2[1], Sim2[0], Sim2[3]]])
    #         if (np.array(np.abs(Sim-Sim2)) < tolerance).all():
    #             temp_consensus_set.append(j)
    #             temp_consensus_set.append(j+1)
    #     if len(temp_consensus_set) > len(consensus_set):
    #         consensus_set = temp_consensus_set
    #         #  best_match = idxs
    #         best_sim = Sim
    # consensus_matches = np.array(matches)[consensus_set]
    # matched_image = np.array([])
    # # draw matches with biggest consensus
    # matched_image = cv2.drawMatches(img1, kpts1, img2, kpts2, consensus_matches[:100],
    #                                 flags=2, outImg=matched_image)
    # plt.figure()
    # plt.imshow(matched_image)
    # plt.show()
    # print('Best match:\nSim=\n%s\n with consensus=%d or %d%%'%(
    #     best_sim, len(consensus_set)/2, 100*len(consensus_set)/2/len(matches)))


    tolerance = 1  # used when comparing similarity matrices of match pairs
    consensus_set = []
    best_sim = []
    # find consensus of translation between a random keypoint and the rest for
    # a number of times to find the best match regarding translation
    for i in range(100):
        idxs = random.sample(range(len(matches)), 3)
        # calc similarity between kp1, kp2 and kp3
        kp11 = kpts1[matches[idxs[0]].queryIdx]
        kp12 = kpts2[matches[idxs[0]].trainIdx]
        kp21 = kpts1[matches[idxs[1]].queryIdx]
        kp22 = kpts2[matches[idxs[1]].trainIdx]
        kp31 = kpts1[matches[idxs[2]].queryIdx]
        kp32 = kpts2[matches[idxs[2]].trainIdx]
        #  A = np.array([[kp11.pt[0], kp11.pt[1], 1, 0, 0, 0],
        #  [0, 0, 0, kp11.pt[0], kp11.pt[1], 1],
        #  [kp21.pt[0], kp21.pt[1], 1, 0, 0, 0],
        #  [0, 0, 0, kp21.pt[0], kp21.pt[1], 1],
        #  [kp31.pt[0], kp31.pt[1], 1, 0, 0, 0],
        #  [0, 0, 0, kp31.pt[0], kp31.pt[1], 1]])
        #  b = np.array([kp12.pt[0], kp12.pt[1], kp22.pt[0], kp22.pt[1],
        #  kp32.pt[0], kp32.pt[1]])
        #  Sim,_,_,_ = np.linalg.lstsq(A, b)
        #  Sim = Sim.reshape((2, 3))
        pts1 = np.float32([[kp11.pt[0], kp11.pt[1]],
                           [kp21.pt[0], kp21.pt[1]],
                           [kp31.pt[0], kp31.pt[1]]])
        pts2 = np.float32([[kp12.pt[0], kp12.pt[1]],
                           [kp22.pt[0], kp22.pt[1]],
                           [kp32.pt[0], kp32.pt[1]]])
        Sim = cv2.getAffineTransform(pts1, pts2)
        temp_consensus_set = []
        for j in range(len(matches) - 2):
            kp11 = kpts1[matches[j].queryIdx]
            kp12 = kpts2[matches[j].trainIdx]
            kp21 = kpts1[matches[j + 1].queryIdx]
            kp22 = kpts2[matches[j + 1].trainIdx]
            kp31 = kpts1[matches[j + 2].queryIdx]
            kp32 = kpts2[matches[j + 2].trainIdx]
            pts1 = np.float32([[kp11.pt[0], kp11.pt[1]],
                               [kp21.pt[0], kp21.pt[1]],
                               [kp31.pt[0], kp31.pt[1]]])
            pts2 = np.float32([[kp12.pt[0], kp12.pt[1]],
                               [kp22.pt[0], kp22.pt[1]],
                               [kp32.pt[0], kp32.pt[1]]])
            Sim2 = cv2.getAffineTransform(pts1, pts2)
            if (np.array(np.abs(Sim - Sim2)) < tolerance).all():
                temp_consensus_set.append(j)
                temp_consensus_set.append(j + 1)
                temp_consensus_set.append(j + 2)
        if len(temp_consensus_set) > len(consensus_set):
            consensus_set = temp_consensus_set
            best_sim = Sim
    consensus_matches = np.array(matches)[consensus_set]
    matched_image = np.array([])
    # draw matches with biggest consensus
    matched_image = cv2.drawMatches(img1, kpts1, img2, kpts2, consensus_matches,
                                    flags=2, outImg=matched_image)
    plt.figure()
    plt.imshow(matched_image)
    plt.show()
    print('Best match:\nSim=\n%s\n with consensus=%d or %d%%'%(
        best_sim, len(consensus_set)/2, 100*len(consensus_set)/2/len(matches)))


    # tolerance = 10
    # best_match = 0
    # consensus_set = []
    # # find consensus of translation between a random keypoint and the rest for
    # # a number of times to find the best match regarding translation
    # for i in range(100):
    #     idx = random.randint(0, len(matches) - 1)
    #     kp1 = kpts1[matches[idx].queryIdx]
    #     kp2 = kpts2[matches[idx].trainIdx]
    #     dx = int(kp1.pt[0] - kp2.pt[0])
    #     dy = int(kp1.pt[1] - kp2.pt[1])
    #     temp_consensus_set = []
    #     for j, match in enumerate(matches):
    #         kp1 = kpts1[match.queryIdx]
    #         kp2 = kpts2[match.trainIdx]
    #         dxi = int(kp1.pt[0] - kp2.pt[0])
    #         dyi = int(kp1.pt[1] - kp2.pt[1])
    #         if abs(dx - dxi) < tolerance and abs(dy - dyi) < tolerance:
    #             temp_consensus_set.append(j)
    #     if len(temp_consensus_set) > len(consensus_set):
    #         consensus_set = temp_consensus_set
    #         best_match = idx
    # # calculate best match translation
    # kp1 = kpts1[matches[best_match].queryIdx]
    # kp2 = kpts2[matches[best_match].trainIdx]
    # dx = int(kp1.pt[0] - kp2.pt[0])
    # dy = int(kp1.pt[1] - kp2.pt[1])
    # consensus_matches = np.array(matches)[consensus_set]
    # matched_image = np.array([])
    # # draw matches with biggest consensus
    # matched_image = cv2.drawMatches(img1, kpts1, img2, kpts2, consensus_matches,
    #                                 flags=2, outImg=matched_image)
    # plt.figure()
    # plt.imshow(matched_image)
    # plt.show()
    # print('Best match: idx=%d with consensus=%d or %d%%\nTranslation: dx=%dpx '
    #       'and dy=%dpx'%(best_match, len(consensus_set),
    #                     100*len(consensus_set) / len(matches), dx, dy))

refname = 'pic/ArtRoom'
retnames = ['_0.75_sc', '_0.75_lg', '_0.75_multiop', '_0.75_qp',
            '_0.75_scl', '_0.75_sm', '_0.75_sns', '_0.75_sv',
            '_0.75_warp', '_0.75_cr']

for retname in retnames:
    retname = refname + retname
    print retname
    img1 = readImg(refname + '.png', 330, 430, 180, 280)    # queryImage
    sal1 = readMat(refname + '.mat', 330, 430, 180, 280)
    print sal1.shape, '-', sal1.dtype, sal1.size

    img2 = readImg(retname + '.png', 280, 480, 130, 330)   # trainImage
    sal2 = readMat(retname + '.mat', 280, 480, 130, 330)
    print sal2.shape, '-', sal2.dtype, sal2.size

    siftMatch(img1, img2)
    cv2.waitKey(0)
    plt.close()
    print
