import torch.utils.data as data
import numpy as np
import os, cv2, random, time, math


def readImg(filename):
    """"
    Color image loaded by OpenCV is in BGR mode, but Matplotlib displays in RGB mode.
    cv2.imread(path, style)
        1 - cv2.IMREAD_COLOR
        0 - cv2.IMREAD_GRAYSCALE
        -1 - cv2.IMREAD_UNCHANGED
    """
    # print filename
    img = cv2.imread(filename, 1)
    return img


def showPatch(n):
    window_name = 'Patch'
    merge_img = np.zeros(tuple([n, n]), 'uint8')
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, merge_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_in_one(src, global_patch, sup, h_w_ratio,method, name, index):
    font = cv2.FONT_HERSHEY_SIMPLEX
    n = 0
    num = 9
    for i in xrange(num):
        if len(global_patch[i]):
            n += 1
    blank_size = 28

    h = 9*blank_size
    w = src[0].shape[1] + blank_size*3
    w_max = 0
    ker_h = []
    for i in xrange(num):
        if len(global_patch[i]):
            h_max = 0
            tmp_w = 0
            for j in xrange(3):
                tmp_h = global_patch[i][j].shape[0]
                tmp_w += global_patch[i][j].shape[1]
                h_max = max(h_max, tmp_h)
            w_max = max(w_max, tmp_w)
            ker_h.append(h_max)
            h+= h_max
    w += w_max
    h = max(h, 2*blank_size+src[0].shape[0]+src[1].shape[0]+src[2].shape[0])

    # print images[0].dtype
    merge_img = np.zeros((h, w, 3), src[0].dtype)

    h_start = 0
    for j in range(3):
        # print count=
        h_end = h_start + src[j].shape[0]
        merge_img[h_start:h_end, 0:src[j].shape[1]] = src[j]
        if j == 1:
            cv2.putText(merge_img, '{}'.format(sup[0]), (3, h_start - 3), font, 0.8, (255,255,255), 1,
                        cv2.LINE_AA)
            cv2.putText(merge_img, method[0], (42, h_start - 3), font, 0.8, (255,255,255), 1,
                        cv2.LINE_AA)
            cv2.putText(merge_img, '{}'.format(index), (240, h_start - 3), font, 0.8, (255,255,255), 1,
                        cv2.LINE_AA)
        elif j == 2:
            cv2.putText(merge_img, '{}'.format(sup[1]), (3, h_start - 3), font, 0.8, (255,255,255), 1,
                        cv2.LINE_AA)
            cv2.putText(merge_img, method[1], (42, h_start - 3), font, 0.8, (255, 255, 255), 1,
                        cv2.LINE_AA)
        h_start = h_end + blank_size

    if n:
        gap = h / n - int(np.mean(ker_h)+1)
        h_start = blank_size
        i = 0
        for k in xrange(num):
            if len(global_patch[k]):
                w_start = src[0].shape[1]+blank_size
                cv2.putText(merge_img, '{}'.format(i), (w_start-24, h_start + 60), font, 0.8, (255,255,255), 1,
                                cv2.LINE_AA)
                for j in xrange(3):
                    if j:
                        cv2.putText(merge_img, h_w_ratio[2*i+j-1][0], (w_start, h_start - 3), font, 0.8, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        cv2.putText(merge_img, h_w_ratio[2*i+j-1][1], (w_start + 108, h_start - 3), font, 0.8, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                    merge_img[h_start:(h_start+global_patch[k][j].shape[0]),
                    w_start:(w_start+global_patch[k][j].shape[1])] = global_patch[k][j]
                    w_start += blank_size+global_patch[k][j].shape[1]
                h_start += gap+ker_h[i]
                i+=1
    cv2.imwrite(name, merge_img)
    # window_name = 'Overall'
    # cv2.namedWindow(window_name)
    # cv2.imshow(window_name, merge_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def show_in_one(images, h_w1, h_w2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    n = len(images)
    blank_size = 28
    window_name = 'Triplet'
    small_h, small_w = images[0].shape[:2]
    for i in xrange(1, 3):
        tmp_h, tmp_w = images[i].shape[:2]
        if tmp_h > small_h:
            small_h = tmp_h
        if tmp_w > small_w:
            small_w = tmp_w
    print n, '-', small_h, small_w,
    small_w = n*(blank_size+small_w)
    merge_img = np.zeros((small_h+blank_size, small_w, 3), images[0].dtype)
    print '-', small_h, small_w

    w_start = 0
    for j in range(n):
        if j == 1:
            cv2.putText(merge_img, '{}'.format(h_w1[0]), (w_start, blank_size - 3), font, 0.3, (255,255,255), 1,
                        cv2.LINE_AA)
            cv2.putText(merge_img, '{}'.format(h_w1[1]), (w_start+36, blank_size - 3), font, 0.3, (255,255,255), 1,
                        cv2.LINE_AA)
        elif j == 2:
            cv2.putText(merge_img, '{}'.format(h_w2[0]), (w_start, blank_size - 3), font, 0.3, (255,255,255), 1,
                        cv2.LINE_AA)
            cv2.putText(merge_img, '{}'.format(h_w2[1]), (w_start+36, blank_size - 3), font, 0.3, (255,255,255), 1,
                        cv2.LINE_AA)
        w_end = w_start + images[j].shape[1]
        print images[j].shape[0], images[j].shape[1]
        merge_img[blank_size:(blank_size+images[j].shape[0]), w_start:w_end] = images[j]
        w_start = (w_end + blank_size)
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 450, 570)
    cv2.imshow(window_name, merge_img)
    cv2.imwrite('ratio.png', merge_img)


class clear_inside():
    def __init__(self, global_patch, horizontal, vertical, t, show):
        self.patch = global_patch
        self.hor = horizontal
        self.ver = vertical

        self.position = [[[[], [], [], []], [[], [], [], []]] for i in xrange(len(self.patch))]

        self.tolerance_w = t[0]
        self.tolerance_h = t[1]

        self.show = show
        self.directions = ['left', 'right', 'up', 'down']

        self.color = [(255,0,0),(0,255,0),(0,0,255)]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.sift = cv2.xfeatures2d.SIFT_create()

    def detectKeypoint(self):
        self.kpts1, des1 = self.sift.detectAndCompute(self.img1, None)
        self.kpts2, des2 = self.sift.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher()
        self.matches = bf.match(des1, des2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)
        print len(self.matches),
        n = min(len(self.matches), 150)
        self.matches = self.matches[:n]

    def checker_one(self, matchA, matchB):
        # print 'In checker_one'
        x1, y1 = self.kpts1[matchA.queryIdx].pt
        m1, n1 = self.kpts2[matchA.trainIdx].pt
        dx = int(x1 - m1)
        dy = int(y1 - n1)

        x2, y2 = self.kpts1[matchB.queryIdx].pt
        m2, n2 = self.kpts2[matchB.trainIdx].pt
        dxi = int(x2 - m2)
        dyi = int(y2 - n2)
        if abs(dx - dxi) < self.tolerance_w and abs(dy - dyi) < self.tolerance_h:
            # print 'True, out checker_one'
            return True
        # print 'False, out checker_one'
        return False

    def checker(self, matchA, matchB):
        # print 'In checker'
        x1, y1 = self.kpts1[matchA.queryIdx].pt
        m1, n1 = self.kpts2[matchA.trainIdx].pt

        x2, y2 = self.kpts1[matchB.queryIdx].pt
        m2, n2 = self.kpts2[matchB.trainIdx].pt
        if (x2 - x1) * (m2 - m1) > 0 and (y2 - y1) * (n2 - n1) > 0:
            # print 'True, out checker'
            return True
        # print 'False, out checker_one'
        return False

    def checker_two(self, matchA, matchB, matchC):
        # print 'In checker_two'
        if self.checker_one(matchA, matchC) is False or self.checker_one(matchB, matchC) is False:
            # print 'False, out checker_two'
            return False
        if self.checker(matchA, matchC) is False or self.checker(matchB, matchC) is False:
            # print 'False, out checker_two'
            return False
        # print 'True, out checker_two'
        return True

    def RANSAC(self, step, index, direction):
        if self.show:
            img3 = cv2.drawMatches(self.img1, self.kpts1,
                                   self.img2, self.kpts2,
                                   self.matches, None, flags=2)

        if len(self.matches) < 20:
            return 0

        consensus_set = []
        for i in range(100):
            temp_consensus_set = []
            idxA = random.randint(0, len(self.matches) - 1)
            temp_consensus_set.append(idxA)
            matchA = self.matches[idxA]
            matchB = matchA
            for j in xrange(len(self.matches)):
                matchB = self.matches[j]
                if self.checker_one(matchA, matchB) and self.checker(matchA, matchB):
                # if self.checker(matchA, matchB):
                    temp_consensus_set.append(j)
                    # print 'Found B!'
                    break
            # print 'Continue...'
            if len(temp_consensus_set) == 2:
                for j, matchC in enumerate(self.matches):
                    if self.checker_two(matchA, matchB, matchC):
                        temp_consensus_set.append(j)
            if len(temp_consensus_set) > len(consensus_set):
                consensus_set = temp_consensus_set
        consensus_matches = np.array(self.matches)[consensus_set]

        n = len(consensus_matches)
        if n < 20:
            return 0

        h_r = []
        w_r = []
        for i in xrange(n-1):
            pi0 = self.kpts1[consensus_matches[i].queryIdx].pt
            pi1 = self.kpts2[consensus_matches[i].trainIdx].pt
            for j in xrange(i+1, n):
                pj0 = self.kpts1[consensus_matches[j].queryIdx].pt
                pj1 = self.kpts2[consensus_matches[j].trainIdx].pt
                if pi0[0] != pj0[0]:
                    w_r.append(abs(pi1[0]-pj1[0])/abs(pi0[0]-pj0[0]))
                if pi0[1] != pj0[1]:
                    h_r.append(abs(pi1[1]-pj1[1])/abs(pi0[1]-pj0[1]))
        h_r = np.median(h_r)
        w_r = np.median(w_r)
        print step, index, self.directions[direction], '- Ratio: ', h_r, w_r

        core1 = np.array([0.0, 0.0])
        core2 = np.array([0.0, 0.0])
        for i in consensus_matches:
            # print self.kpts1[i.queryIdx].pt
            # print self.kpts2[i.trainIdx].pt
            core1[0] += (self.kpts1[i.queryIdx].pt)[0]
            core1[1] += (self.kpts1[i.queryIdx].pt)[1]
            core2[0] += (self.kpts2[i.trainIdx].pt)[0]
            core2[1] += (self.kpts2[i.trainIdx].pt)[1]
        core1 /= n
        core2 /= n
        # print core1
        # print core2
        core1[0] = core1[0]*w_r
        core1[1] = core1[1]*h_r
        core2 -= core1
        # print core2
        x = int(core2[0])
        y = int(core2[1])

        scale_w = int(self.img1.shape[1]*w_r) + x
        scale_h = int(self.img1.shape[0]*h_r) + y

        y = max(y, 0)
        x = max(x, 0)
        self.position[step][index][direction] = [y, scale_h, x, scale_w]

        if self.show:
            matchName = "Match{}_{}".format(index, self.directions[direction])
            cv2.namedWindow(matchName)
            cv2.moveWindow(matchName, index*150, 50)
            cv2.imshow(matchName, img3)
            # self.triplet.append(self.img2[y:scale_h, x:scale_w])
            matched_image = np.array([])
            matched_image = cv2.drawMatches(self.img1, self.kpts1,
                                            self.img2, self.kpts2, consensus_matches,
                                            flags=2, outImg=matched_image)
            ransacName = 'Ransac{}_{}'.format(index, self.directions[direction])
            cv2.namedWindow(ransacName)
            cv2.moveWindow(ransacName, index*150, 300)
            cv2.imshow(ransacName, matched_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def generate(self, step, index, direction):
        self.detectKeypoint()
        self.RANSAC(step, index, direction)

    def getContainer(self):
        for i in xrange(9):
            if len(self.patch[i]):
                y = i/3
                x = i%3
                # left - x_end
                self.img1 = self.hor[y][x]
                self.img2 = self.patch[i][1]
                self.generate(i, 0, 0)

                self.img2 = self.patch[i][2]
                self.generate(i, 1, 0)
                # right - x_start
                self.img1 = self.hor[y][x+1]
                self.img2 = self.patch[i][1]
                self.generate(i, 0, 1)

                self.img2 = self.patch[i][2]
                self.generate(i, 1, 1)
                # up - y_end
                self.img1 = self.ver[y][x]
                self.img2 = self.patch[i][1]
                self.generate(i, 0, 2)

                self.img2 = self.patch[i][2]
                self.generate(i, 1, 2)
                # down - y_start
                self.img1 = self.ver[y+1][x]
                self.img2 = self.patch[i][1]
                self.generate(i, 0, 3)

                self.img2 = self.patch[i][2]
                self.generate(i, 1, 3)

        for i in xrange(9):
            if len(self.patch[i]):
                print '********', i, '********'
                # print 'self.position:', self.position[i]
                for j in xrange(2):
                    y_start = 0
                    x_start = 0
                    y_end, x_end, _ = self.patch[i][j+1].shape
                    print y_start, y_end, x_start, x_end, '->',
                    pos = [-1 for k in xrange(4)]
                    # print self.position[i][j]
                    for k in xrange(4):
                        if len(self.position[i][j][k]):
                            pos[k] = self.position[i][j][k][(k+2)%4]
                    if pos[0] != -1:
                        # if pos[0] < x_end/2:
                        x_start = max(pos[0], x_start)
                    if pos[1] != -1:
                        # if pos[1] > x_end/2:
                        x_end = min(pos[1], x_end)
                    if pos[2] != -1:
                        # if pos[2] < y_end/2:
                        y_start = max(pos[2], y_start)
                    if pos[3] != -1:
                        # if pos[3] > y_end/2:
                        y_end = min(pos[3], y_end)
                    print y_start, y_end, x_start, x_end
                    self.patch[i][j+1] = self.patch[i][j+1][y_start:y_end, x_start:x_end]
                    # print '     -------'
                print
        return self.patch


class clear_outside():
    def __init__(self, global_patch, horizontal, vertical, t, show):
        self.patch = global_patch
        self.hor = horizontal
        self.ver = vertical

        self.position = [[[[], [], [], []], [[], [], [], []]] for i in xrange(len(self.patch))]

        self.tolerance_w = t[0]
        self.tolerance_h = t[1]

        self.show = show
        self.directions = ['left', 'right', 'up', 'down']

        self.color = [(255,0,0),(0,255,0),(0,0,255)]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.sift = cv2.xfeatures2d.SIFT_create()

    def detectKeypoint(self):
        print self.img1.shape, self.img2.shape
        if self.img1.shape[0]*self.img1.shape[1] == 0:
            self.matches = []
            return
        if self.img2.shape[0]*self.img2.shape[1] == 0:
            self.matches = []
            return
        self.kpts1, des1 = self.sift.detectAndCompute(self.img1, None)
        self.kpts2, des2 = self.sift.detectAndCompute(self.img2, None)
        if des1 is None or des2 is None:
            self.matches = []
            return
        bf = cv2.BFMatcher()
        self.matches = bf.match(des1, des2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)
        print len(self.matches),
        n = min(len(self.matches), 150)
        self.matches = self.matches[:n]

    def checker_one(self, matchA, matchB):
        # print 'In checker_one'
        x1, y1 = self.kpts1[matchA.queryIdx].pt
        m1, n1 = self.kpts2[matchA.trainIdx].pt
        dx = int(x1 - m1)
        dy = int(y1 - n1)

        x2, y2 = self.kpts1[matchB.queryIdx].pt
        m2, n2 = self.kpts2[matchB.trainIdx].pt
        dxi = int(x2 - m2)
        dyi = int(y2 - n2)
        if abs(dx - dxi) < self.tolerance_w and abs(dy - dyi) < self.tolerance_h:
            # print 'True, out checker_one'
            return True
        # print 'False, out checker_one'
        return False

    def checker(self, matchA, matchB):
        # print 'In checker'
        x1, y1 = self.kpts1[matchA.queryIdx].pt
        m1, n1 = self.kpts2[matchA.trainIdx].pt

        x2, y2 = self.kpts1[matchB.queryIdx].pt
        m2, n2 = self.kpts2[matchB.trainIdx].pt
        if (x2 - x1) * (m2 - m1) > 0 and (y2 - y1) * (n2 - n1) > 0:
            # print 'True, out checker'
            return True
        # print 'False, out checker_one'
        return False

    def checker_two(self, matchA, matchB, matchC):
        # print 'In checker_two'
        if self.checker_one(matchA, matchC) is False or self.checker_one(matchB, matchC) is False:
            # print 'False, out checker_two'
            return False
        if self.checker(matchA, matchC) is False or self.checker(matchB, matchC) is False:
            # print 'False, out checker_two'
            return False
        # print 'True, out checker_two'
        return True

    def RANSAC(self, step, index, direction):
        if self.show:
            img3 = cv2.drawMatches(self.img1, self.kpts1,
                                   self.img2, self.kpts2,
                                   self.matches, None, flags=2)

        if len(self.matches) < 15:
            return 0

        consensus_set = []
        for i in range(100):
            temp_consensus_set = []
            idxA = random.randint(0, len(self.matches) - 1)
            temp_consensus_set.append(idxA)
            matchA = self.matches[idxA]
            matchB = matchA
            for j in xrange(len(self.matches)):
                matchB = self.matches[j]
                if self.checker_one(matchA, matchB) and self.checker(matchA, matchB):
                # if self.checker(matchA, matchB):
                    temp_consensus_set.append(j)
                    # print 'Found B!'
                    break
            # print 'Continue...'
            if len(temp_consensus_set) == 2:
                for j, matchC in enumerate(self.matches):
                    if self.checker_two(matchA, matchB, matchC):
                        temp_consensus_set.append(j)
            if len(temp_consensus_set) > len(consensus_set):
                consensus_set = temp_consensus_set
        consensus_matches = np.array(self.matches)[consensus_set]

        n = len(consensus_matches)
        if n < 15:
            return 0

        h_r = []
        w_r = []
        for i in xrange(n-1):
            pi0 = self.kpts1[consensus_matches[i].queryIdx].pt
            pi1 = self.kpts2[consensus_matches[i].trainIdx].pt
            for j in xrange(i+1, n):
                pj0 = self.kpts1[consensus_matches[j].queryIdx].pt
                pj1 = self.kpts2[consensus_matches[j].trainIdx].pt
                if pi0[0] != pj0[0]:
                    w_r.append(abs(pi1[0]-pj1[0])/abs(pi0[0]-pj0[0]))
                if pi0[1] != pj0[1]:
                    h_r.append(abs(pi1[1]-pj1[1])/abs(pi0[1]-pj0[1]))
        h_r = np.median(h_r)
        w_r = np.median(w_r)
        print step, index, self.directions[direction], '- Ratio: ', h_r, w_r

        core1 = np.array([0.0, 0.0])
        core2 = np.array([0.0, 0.0])
        for i in consensus_matches:
            # print self.kpts1[i.queryIdx].pt
            # print self.kpts2[i.trainIdx].pt
            core1[0] += (self.kpts1[i.queryIdx].pt)[0]
            core1[1] += (self.kpts1[i.queryIdx].pt)[1]
            core2[0] += (self.kpts2[i.trainIdx].pt)[0]
            core2[1] += (self.kpts2[i.trainIdx].pt)[1]
        core1 /= n
        core2 /= n
        # print core1
        # print core2
        core1[0] = core1[0]*w_r
        core1[1] = core1[1]*h_r
        core2 -= core1
        # print core2
        x = int(core2[0])
        y = int(core2[1])

        scale_w = int(self.img1.shape[1]*w_r) + x
        scale_h = int(self.img1.shape[0]*h_r) + y

        y = max(y, 0)
        x = max(x, 0)
        self.position[step][index][direction] = [y, scale_h, x, scale_w]

        # update the patch
        y_start = 0
        x_start = 0
        y_end, x_end, _ = self.patch[step][index+1].shape
        print y_start, y_end, x_start, x_end, '->',
        pos = [-1 for k in xrange(4)]
        pos[direction] = [y, scale_h, x, scale_w][3-direction]
        if pos[0] != -1:
            # if pos[0] < x_end/2:
            x_start = max(pos[0], x_start)
        if pos[1] != -1:
            # if pos[1] > x_end/2:
            x_end = min(pos[1], x_end)
        if pos[2] != -1:
            # if pos[2] < y_end/2:
            y_start = max(pos[2], y_start)
        if pos[3] != -1:
            # if pos[3] > y_end/2:
            y_end = min(pos[3], y_end)
        print y_start, y_end, x_start, x_end
        print step, index

        height, width, _ = self.patch[step][index+1].shape
        height *= 1.0
        width *= 1.0
        ratio = 0.7
        if ((y_end - y_start) / height) > ratio and ((x_end-x_start) / width) > ratio:
            self.patch[step][index + 1] = self.patch[step][index + 1][y_start:y_end, x_start:x_end]
        else:
            self.patch[step][index + 1] = self.patch[step][index + 1][y_start:(y_start+1), x_start:(x_start+1)]
        # print '     -------'

        if self.show:
            matchName = "Match{}_{}".format(index, self.directions[direction])
            cv2.namedWindow(matchName)
            cv2.moveWindow(matchName, index*150, 50)
            cv2.imshow(matchName, img3)
            # self.triplet.append(self.img2[y:scale_h, x:scale_w])
            matched_image = np.array([])
            matched_image = cv2.drawMatches(self.img1, self.kpts1,
                                            self.img2, self.kpts2, consensus_matches,
                                            flags=2, outImg=matched_image)
            ransacName = 'Ransac{}_{}'.format(index, self.directions[direction])
            cv2.namedWindow(ransacName)
            cv2.moveWindow(ransacName, index*150, 300)
            cv2.imshow(ransacName, matched_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def generate(self, step, index, direction):
        self.detectKeypoint()
        self.RANSAC(step, index, direction)

    def getContainer(self):
        for i in xrange(9):
            if len(self.patch[i]):
                y = i/3
                x = i%3
                if x > 0: # left - x_end
                    for j in xrange(x):
                        self.img1 = self.hor[y][j]
                        self.img2 = self.patch[i][1]
                        self.generate(i, 0, 0)

                        self.img2 = self.patch[i][2]
                        self.generate(i, 1, 0)
                if x < 2: # right - x_start
                    for j in xrange(3, x+1, -1):
                        self.img1 = self.hor[y][j]
                        self.img2 = self.patch[i][1]
                        self.generate(i, 0, 1)

                        self.img2 = self.patch[i][2]
                        self.generate(i, 1, 1)
                if y > 0: # up - y_end
                    for j in xrange(y):
                        self.img1 = self.ver[j][x]
                        self.img2 = self.patch[i][1]
                        self.generate(i, 0, 2)

                        self.img2 = self.patch[i][2]
                        self.generate(i, 1, 2)
                if y < 2: # down - y_start
                    for j in xrange(3, y+1, -1):
                        self.img1 = self.ver[j][x]
                        self.img2 = self.patch[i][1]
                        self.generate(i, 0, 3)

                        self.img2 = self.patch[i][2]
                        self.generate(i, 1, 3)
        #
        # for i in xrange(9):
        #     if len(self.patch[i]):
        #         print '********', i, '********'
        #         # print 'self.position:', self.position[i]
        #         for j in xrange(2):
        #             y_start = 0
        #             x_start = 0
        #             y_end, x_end, _ = self.patch[i][j+1].shape
        #             print y_start, y_end, x_start, x_end, '->',
        #             pos = [-1 for k in xrange(4)]
        #             # print self.position[i][j]
        #             for k in xrange(4):
        #                 if len(self.position[i][j][k]):
        #                     pos[k] = self.position[i][j][k][3-k]
        #             if pos[0] != -1:
        #                 # if pos[0] < x_end/2:
        #                 x_start = max(pos[0], x_start)
        #             if pos[1] != -1:
        #                 # if pos[1] > x_end/2:
        #                 x_end = min(pos[1], x_end)
        #             if pos[2] != -1:
        #                 # if pos[2] < y_end/2:
        #                 y_start = max(pos[2], y_start)
        #             if pos[3] != -1:
        #                 # if pos[3] > y_end/2:
        #                 y_end = min(pos[3], y_end)
        #             print y_start, y_end, x_start, x_end
        #             self.patch[i][j+1] = self.patch[i][j+1][y_start:y_end, x_start:x_end]
        #             # print '     -------'
        #         print
        container = []
        for i in xrange(len(self.patch)):
            tmp = []
            for j in xrange(len(self.patch[i])):
                if self.patch[i][j].shape[0] >= 64 and self.patch[i][j].shape[1] >= 64:
                    tmp.append(self.patch[i][j])
            if len(tmp) > 2:
                container.append(tmp)
            else:
                container.append([])
        return container


class patch():
    def __init__(self, show, output, tag=0):
        self.show = show
        self.output = output
        self.tag = tag # 0 - h_r and w_r equal 1.0, otherwise local patch determines
        self.imgDir = 'dataset/'
        self.outDir = 'global_patch/'
        if os.path.exists(self.outDir) == False:
            os.mkdir(self.outDir)
        self.overDir = 'global_patch_over/'
        if os.path.exists(self.overDir) == False:
            os.mkdir(self.overDir)
        self.name = ['sc', 'sc']
        self.color = [(255,0,0),(0,255,0),(0,0,255)]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.sift = cv2.xfeatures2d.SIFT_create()

    def setN(self, refname, retname1, retname2, outname, sup, step):
        self.step = step
        print refname, retname1, retname2, sup
        self.iter = 0
        self.filename = refname
        self.outName = outname
        self.sup = sup

        method = retname1.split('_')
        self.name[0] = method[-1]
        method = retname2.split('_')
        self.name[1] = method[-1]
        print self.name, outname

        basisDir = self.imgDir + refname + '/'
        self.src1 = readImg(basisDir + refname + '.png')  # queryImage
        self.draw1 = self.src1.copy()
        self.src2 = readImg(basisDir + retname1 + '.png')  # trainImage
        self.draw2 = self.src2.copy()
        self.src3 = readImg(basisDir + retname2 + '.png')  # trainImage
        self.draw3 = self.src3.copy()

        self.h = self.src1.shape[0]  # the height of reference image
        self.w = self.src1.shape[1]  # the width of reference image
        # print self.src1.shape
        self.h_n = 3
        self.w_n = 3

        # self.tolerance_h = max(20, (self.h - self.src2.shape[0])/4)
        # self.tolerance_w = max(20, (self.w - self.src2.shape[1])/4)
        self.tolerance_h = 20
        self.tolerance_w = 20
        print self.tolerance_h, self.tolerance_w

        self.ker_h = self.h/2
        self.ker_w = self.w/2
        self.stride_h = self.ker_h / 2
        self.stride_w = self.ker_w / 2

        self.h_basis = (self.h % self.ker_h ) / 2
        self.w_basis = (self.w % self.ker_w) / 2

        self.x = 0
        self.y = 0
        self.counter = 0
        self.h_w_ratio = []
        self.global_patch = []
        self.hor = [[] for i in xrange(3)]
        self.ver = [[] for i in xrange(4)]
        self.center_container = []
        self.center_postion = [[0, 0], [0, 0]]
        self.position_container = []
        self.position = [[0, 0], [0, 0], [0, 0]]

    def detectKeypoint(self):
        self.kpts1, des1 = self.sift.detectAndCompute(self.img1, None)
        self.kpts2, des2 = self.sift.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher()
        self.matches = bf.match(des1, des2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)
        print len(self.matches),
        n = min(len(self.matches), 150)
        self.matches = self.matches[:n]

    def checker_one(self, matchA, matchB):
        # print 'In checker_one'
        x1, y1 = self.kpts1[matchA.queryIdx].pt
        m1, n1 = self.kpts2[matchA.trainIdx].pt
        dx = int(x1 - m1)
        dy = int(y1 - n1)

        x2, y2 = self.kpts1[matchB.queryIdx].pt
        m2, n2 = self.kpts2[matchB.trainIdx].pt
        dxi = int(x2 - m2)
        dyi = int(y2 - n2)
        if abs(dx - dxi) < self.tolerance_w and abs(dy - dyi) < self.tolerance_h:
            # print 'True, out checker_one'
            return True
        # print 'False, out checker_one'
        return False

    def checker(self, matchA, matchB):
        # print 'In checker'
        x1, y1 = self.kpts1[matchA.queryIdx].pt
        m1, n1 = self.kpts2[matchA.trainIdx].pt

        x2, y2 = self.kpts1[matchB.queryIdx].pt
        m2, n2 = self.kpts2[matchB.trainIdx].pt
        if (x2 - x1) * (m2 - m1) > 0 and (y2 - y1) * (n2 - n1) > 0:
            # print 'True, out checker'
            return True
        # print 'False, out checker_one'
        return False

    def checker_two(self, matchA, matchB, matchC):
        # print 'In checker_two'
        if self.checker_one(matchA, matchC) is False or self.checker_one(matchB, matchC) is False:
            # print 'False, out checker_two'
            return False
        if self.checker(matchA, matchC) is False or self.checker(matchB, matchC) is False:
            # print 'False, out checker_two'
            return False
        # print 'True, out checker_two'
        return True

    def RANSAC(self, index):
        if self.show:
            img3 = cv2.drawMatches(self.img1, self.kpts1,
                                   self.img2, self.kpts2,
                                   self.matches, None, flags=2)
            matchName = "Match{}".format(index)
            cv2.namedWindow(matchName)
            cv2.moveWindow(matchName, index*150, 50)
            cv2.imshow(matchName, img3)

        if len(self.matches) < 3:
            return 0

        consensus_set = []
        for i in range(100):
            temp_consensus_set = []
            idxA = random.randint(0, len(self.matches) - 1)
            temp_consensus_set.append(idxA)
            matchA = self.matches[idxA]
            matchB = matchA
            for j in xrange(len(self.matches)):
                matchB = self.matches[j]
                if self.checker_one(matchA, matchB) and self.checker(matchA, matchB):
                # if self.checker(matchA, matchB):
                    temp_consensus_set.append(j)
                    # print 'Found B!'
                    break
            # print 'Continue...'
            if len(temp_consensus_set) == 2:
                for j, matchC in enumerate(self.matches):
                    if self.checker_two(matchA, matchB, matchC):
                        temp_consensus_set.append(j)
            if len(temp_consensus_set) > len(consensus_set):
                consensus_set = temp_consensus_set
        consensus_matches = np.array(self.matches)[consensus_set]

        n = len(consensus_matches)
        if n < 3:
            return 0

        h_r = []
        w_r = []
        # num = (n-1)*n/2
        # num_h = num
        # num_w = num_h
        for i in xrange(n-1):
            pi0 = self.kpts1[consensus_matches[i].queryIdx].pt
            pi1 = self.kpts2[consensus_matches[i].trainIdx].pt
            # print pi0, pi1
            for j in xrange(i+1, n):
                pj0 = self.kpts1[consensus_matches[j].queryIdx].pt
                pj1 = self.kpts2[consensus_matches[j].trainIdx].pt
                # print pj0, pj1
                if pi0[0] != pj0[0]:
                    w_r.append(abs(pi1[0]-pj1[0])/abs(pi0[0]-pj0[0]))
                if pi0[1] != pj0[1]:
                    h_r.append(abs(pi1[1]-pj1[1])/abs(pi0[1]-pj0[1]))
        h_r = np.median(h_r)
        w_r = np.median(w_r)
        self.h_w_ratio.append([format(h_r, '.3f'), format(w_r, '.3f')])
        print self.counter, '- Ratio: ', h_r, w_r

        core1 = np.array([0.0, 0.0])
        core2 = np.array([0.0, 0.0])
        for i in consensus_matches:
            # print self.kpts1[i.queryIdx].pt
            # print self.kpts2[i.trainIdx].pt
            core1[0] += (self.kpts1[i.queryIdx].pt)[0]
            core1[1] += (self.kpts1[i.queryIdx].pt)[1]
            core2[0] += (self.kpts2[i.trainIdx].pt)[0]
            core2[1] += (self.kpts2[i.trainIdx].pt)[1]
        core1 /= n
        core2 /= n
        # print core1
        # print core2
        if self.tag:
            core1[0] = core1[0]*w_r
            core1[1] = core1[1]*h_r
        core2 -= core1
        # print core2
        x = int(core2[0])
        y = int(core2[1])
        # print y, x, self.ker_w*w_r, self.ker_h*h_r
        self.center_postion[index-1] = [y, x]
        scale_w = self.ker_w
        scale_h = self.ker_h
        if self.tag:
            scale_w = int(scale_w*w_r) + x
            scale_h = int(scale_h*h_r) + y
        else:
            scale_w = self.ker_w + x
            scale_h = self.ker_h + y
        self.scale.append((scale_h, scale_w))
        y = max(y, 0)
        x = max(x, 0)
        self.double.append((y, x))
        self.position[index] = [y, scale_h, x, scale_w]

        if self.show:
            self.triplet.append(self.img2[y:scale_h, x:scale_w])
            matched_image = np.array([])
            matched_image = cv2.drawMatches(self.img1, self.kpts1,
                                            self.img2, self.kpts2, consensus_matches,
                                            flags=2, outImg=matched_image)
            ransacName = 'Ransac{}'.format(index)
            cv2.namedWindow(ransacName)
            cv2.moveWindow(ransacName, index*150, 300)
            cv2.imshow(ransacName, matched_image)

    def generate(self, index):
        self.detectKeypoint()
        self.RANSAC(index)

    def match(self):
        self.img2 = self.src2
        self.generate(1)

        self.img2 = self.src3
        self.generate(2)

    def getTriplet(self):
        for i in xrange(0, 2):
            y = self.double[i][0]
            x = self.double[i][1]
            if i == 0:
                self.ret1 = self.src2[y:self.scale[i][0], x:self.scale[i][1]]
                cv2.rectangle(self.draw2, (x, y), (self.scale[i][1], self.scale[i][0]), self.color[self.counter%3], 1)
                cv2.putText(self.draw2, '{}'.format(self.counter), (x, y+21), self.font, 0.8,self.color[self.counter%3]
                            , 1,cv2.LINE_AA)
            else:
                self.ret2 = self.src3[y:self.scale[i][0], x:self.scale[i][1]]
                cv2.rectangle(self.draw3, (x, y), (self.scale[i][1], self.scale[i][0]), self.color[self.counter%3], 1)
                cv2.putText(self.draw3, '{}'.format(self.counter), (x, y+21), self.font, 0.8,self.color[self.counter%3]
                            , 1,cv2.LINE_AA)

    def update(self):
        if self.x + 1 < self.w_n:
            self.x += 1
        else:
            self.y += 1
            self.x = 0
        if self.y >= self.h_n:
            self.iter = 1

    def showHorVer(self):
        for i in xrange(3):
            for j in xrange(len(self.hor[i])):
                name = 'hor{}_{}'.format(i, j)
                cv2.namedWindow(name)
                cv2.imshow(name, self.hor[i][j])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for i in xrange(4):
            for j in xrange(len(self.ver[i])):
                name = 'ver{}_{}'.format(i, j)
                cv2.namedWindow(name)
                cv2.imshow(name, self.ver[i][j])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def getNext(self):
        if self.iter:
            src = [self.draw1, self.draw2, self.draw3]
            draw_in_one(src, self.global_patch, self.sup, self.h_w_ratio, self.name,
                        self.overDir + self.filename+'_over.png', self.step)

            cleaner = clear_inside(self.global_patch, self.hor, self.ver,
                            (self.tolerance_w, self.tolerance_h), 0)
            container = cleaner.getContainer()
            draw_in_one(src, container, self.sup, self.h_w_ratio, self.name,
                        self.overDir + self.filename+'_over_clear1.png', self.step)

            cleaner = clear_outside(container, self.hor, self.ver,
                            (self.tolerance_w, self.tolerance_h), 0)
            container = cleaner.getContainer()
            draw_in_one(src, container, self.sup, self.h_w_ratio, self.name,
                        self.overDir + self.filename+'_over_clear2.png', self.step)
            print '    The total number of triplet: {}'.format(self.counter)
            return 0
        y_ = self.h_basis + self.stride_h * self.y
        x_ = self.w_basis + self.stride_w * self.x
        self.hor[self.y].append(self.src1[y_:(y_ + self.ker_h), x_:(x_ + self.stride_w)])
        if self.x == 2:
            self.hor[self.y].append(self.src1[y_:(y_ + self.ker_h), (x_+self.stride_w):(x_ + self.ker_w)])
        self.ver[self.y].append(self.src1[y_:(y_ + self.stride_h), x_:(x_ + self.ker_w)])
        if self.y == 2:
            self.ver[self.y+1].append(self.src1[(y_ + self.stride_h):(y_ + self.ker_h), x_:(x_ + self.ker_w)])

        self.img1 = self.src1[y_:(y_ + self.ker_h), x_:(x_ + self.ker_w)]
        self.ref = self.img1
        self.position[0] = [y_, y_ + self.ker_h, x_, x_ + self.ker_w]

        cv2.rectangle(self.draw1, (x_, y_), (x_ + self.ker_w, y_ + self.ker_h), self.color[self.counter%3], 1)
        cv2.putText(self.draw1, '{}'.format(self.counter), (x_, y_+21), self.font, 0.8,self.color[self.counter%3]
                    , 1,cv2.LINE_AA)

        self.update()

        if self.show:
            self.triplet = [self.img1]

        self.double = []
        self.scale = []

        if self.show:
            cv2.namedWindow("Crop")
            cv2.moveWindow("Crop", 10, 0)
            cv2.imshow('Crop', self.img1)

        self.match()

        if len(self.double) == 2:
            self.getTriplet()
            if self.ref.shape[0] < 64 or self.ref.shape[1] < 64:
                self.global_patch.append([])
                return 1
            if self.ret1.shape[0] < 64 or self.ret1.shape[1] < 64:
                self.global_patch.append([])
                return 1
            if self.ret2.shape[0] < 64 or self.ret2.shape[1] < 64:
                self.global_patch.append([])
                return 1
            tmp = []
            for i in xrange(3):
                tmp.append(self.position[i])
            self.position_container.append(tmp)
            tmp = []
            for i in xrange(2):
                tmp.append(self.center_postion[i])
            self.center_container.append(tmp)
            name = self.outDir + self.outName
            self.global_patch.append([self.ref,self.ret1,self.ret2])
            if self.output:
                self.saveImg(name + '{}_{}.bmp'.format(self.counter, 0), self.ref)
                self.saveImg(name + '{}_{}.bmp'.format(self.counter, 1), self.ret1)
                self.saveImg(name + '{}_{}.bmp'.format(self.counter, 2), self.ret2)
            self.counter += 1

            if self.show:
                show_in_one(self.triplet, self.h_w_ratio[2*(self.counter-1)],
                            self.h_w_ratio[2*(self.counter-1)+1])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return 2
        self.global_patch.append([])
        if self.show:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 1

    def saveImg(self, outname, img):
        cv2.imwrite(outname, img)


class generator(data.Dataset):
    def __init__(self, show, output, tag):
        super(generator, self).__init__()
        self.disType = []
        self.filenames = []
        self.scores = []
        self.loadFile()

        self.tag = 3
        self.container = []
        self.triplet = patch(show, output, tag)

        self.txtName = 'GT/list.txt'
        out = open(self.txtName, 'w')
        out.close()

    def loadFile(self):
        f = open('GT/subjRef.txt')
        line = f.readline().strip()
        for part in line.split(' '):
            self.disType.append(part)
        for line in f.readlines():
            tmp = line.split(' ')
            name = tmp[2]

            helpt = []
            for i in xrange(4, len(tmp)):
                helpt.append(int(tmp[i]))

            for i in xrange(7):
                dis1 = self.disType[i]
                for j in xrange(i + 1, 8):
                    dis2 = self.disType[j]
                    self.filenames.append([name, name + '_' + tmp[3] + '_' + dis1,
                                           name + '_' + tmp[3] + '_' + dis2,
                                           '{}_{}_{}_'.format(name, dis1, dis2)])
                    self.scores.append([helpt[i], helpt[j]])
        f.close()

    def show(self):
        cv2.namedWindow('ref')
        cv2.moveWindow("ref", 10, 30)
        cv2.imshow('ref', self.triplet.ref)
        cv2.namedWindow("ret1")
        cv2.moveWindow("ret1", 10, 230)
        cv2.imshow('ret1', self.triplet.ret1)
        cv2.namedWindow("ret2")
        cv2.moveWindow("ret2", 10, 430)
        cv2.imshow('ret2', self.triplet.ret2)
        cv2.waitKey(0)

    def __getitem__(self, index):
        if self.tag == 3:
            self.triplet.setN(self.filenames[index][0],
                              self.filenames[index][1],
                              self.filenames[index][2],
                              self.filenames[index][3],
                              self.scores[index], index)
            self.tag = 2
            return
        while self.tag:
            self.tag = self.triplet.getNext()
            if self.tag == 2:
                # self.show()
                return
            elif self.tag == 0:
                self.tag = 3
                if self.triplet.counter < 9:
                    self.container.append([self.triplet.counter, self.filenames[index][3]])
                out = open(self.txtName, 'a')
                print self.filenames[index][3]
                print >> out, self.filenames[index][3], self.triplet.counter
                out.close()
                return

    def __len__(self):
        return len(self.filenames)

    def statistic(self):
        out = open('GT/less9.txt', 'w')
        print >> out, len(self.container)
        for i in self.container:
            print >> out, i
        out.close()


index = 1
data = generator(show = 0, output=0, tag=1)

if index == 0:
    start = time.time()
    for i in xrange(0, 1):
        print '     {}'.format(i)
        s = time.time()
        index = 362
        data[index]
        while data.tag == 2:
            data[index]
        print '    Step consuming: {}'.format(time.time()-s)
    print '    Time consuming: {}'.format(time.time()-start)
elif index == 1:
    start = time.time()
    con = []
    for i in xrange(37):
        con.append(i)
    random.shuffle(con)
    print con

    for j in xrange(0, 37):
        print '     {}'.format(j)
        s = time.time()
        if j < 28:
            index = j
        else:
            index = random.randint(0, 27)
        i = con[j]
        data[i*28+index]
        while data.tag == 2:
            data[i*28+index]
        print index
        print '    Step consuming: {}'.format(time.time()-s)
    print '    Time consuming: {}'.format(time.time()-start)
else:
    start = time.time()
    for i in xrange(37*28):
        s = time.time()
        data[i]
        while data.tag == 2:
            data[i]
        print '    Step consuming: {}'.format(time.time()-s)
    data[0]
    print '    Time consuming: {}'.format(time.time()-start)
data.statistic()