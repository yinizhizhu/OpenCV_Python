import torch.utils.data as data
import numpy as np
import os, cv2, random, time
import scipy.io as scio


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


def show_in_one(images):
    n = len(images)
    blank_size = 6
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
    merge_img = np.zeros((small_h, small_w, 3), images[0].dtype)
    print '-', small_h, small_w

    w_start = 0
    for j in range(n):
        w_end = w_start + images[j].shape[1]
        print images[j].shape[0], images[j].shape[1]
        merge_img[0:images[j].shape[0], w_start:w_end] = images[j]
        w_start = (w_end + blank_size)
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 450, 570)
    cv2.imshow(window_name, merge_img)
    cv2.imwrite('ratio.png', merge_img)


class patch():
    def __init__(self, show, tag=0):
        self.show = show    # 1 - show each step of selection and result of matching
        self.tag = tag # 0 - h_r and w_r equal 1.0, otherwise local patch determines
        self.imgDir = 'dataset/'
        self.filename = ''
        self.sift = cv2.xfeatures2d.SIFT_create()

    def setN(self, refname, retname1, retname2, outname):
        self.iter = 0
        self.outName = outname

        basisDir = self.imgDir + refname + '/'
        self.src1 = readImg(basisDir + refname + '.png')  # queryImage
        self.src2 = readImg(basisDir + retname1 + '.png')  # trainImage
        self.src3 = readImg(basisDir + retname2 + '.png')  # trainImage

        self.h = self.src1.shape[0]  # the height of reference image
        self.w = self.src1.shape[1]  # the width of reference image
        # print self.src1.shape
        self.h_n = 3
        self.w_n = 3

        self.ker_h = self.h/2
        self.ker_w = self.w/2
        self.stride_h = self.ker_h / 2
        self.stride_w = self.ker_w / 2

        self.h_basis = (self.h % self.ker_h ) / 2
        self.w_basis = (self.w % self.ker_w) / 2

        self.x = 0
        self.y = 0
        self.h_r = 1.0
        self.w_r = 1.0
        self.counter = 0

    def detectKeypoint(self):
        self.kpts1, des1 = self.sift.detectAndCompute(self.img1, None)
        self.kpts2, des2 = self.sift.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher()
        self.matches = bf.match(des1, des2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)

    def RANSAC(self, index):
        img3 = cv2.drawMatches(self.img1, self.kpts1,
                               self.img2, self.kpts2,
                               self.matches, None, flags=2)
        if self.show:
            matchName = "Match{}".format(index)
            cv2.namedWindow(matchName)
            cv2.moveWindow(matchName, index*150, 50)
            cv2.imshow(matchName, img3)

        tolerance = 10
        consensus_set = []
        if len(self.matches) < 3:
            return 0

        for i in range(100):
            idx = random.randint(0, len(self.matches) - 1)
            kp1 = self.kpts1[self.matches[idx].queryIdx]
            kp2 = self.kpts2[self.matches[idx].trainIdx]
            dx = int(kp1.pt[0] - kp2.pt[0])
            dy = int(kp1.pt[1] - kp2.pt[1])
            temp_consensus_set = []
            for j, match in enumerate(self.matches):
                kp1 = self.kpts1[match.queryIdx]
                kp2 = self.kpts2[match.trainIdx]
                dxi = int(kp1.pt[0] - kp2.pt[0])
                dyi = int(kp1.pt[1] - kp2.pt[1])
                if abs(dx - dxi) < tolerance and abs(dy - dyi) < tolerance:
                    temp_consensus_set.append(j)
            if len(temp_consensus_set) > len(consensus_set):
                consensus_set = temp_consensus_set
        consensus_matches = np.array(self.matches)[consensus_set]
        matched_image = np.array([])
        matched_image = cv2.drawMatches(self.img1, self.kpts1,
                                        self.img2, self.kpts2, consensus_matches,
                                        flags=2, outImg=matched_image)

        n = len(consensus_matches)
        if n < 3:
            return 0

        h_r = 0.0
        w_r = 0.0
        for i in xrange(n-1):
            pi0 = self.kpts1[consensus_matches[i].queryIdx].pt
            pi1 = self.kpts2[consensus_matches[i].trainIdx].pt
            # print pi0, pi1
            for j in xrange(i+1, n):
                pj0 = self.kpts1[consensus_matches[j].queryIdx].pt
                pj1 = self.kpts2[consensus_matches[j].trainIdx].pt
                # print pj0, pj1
                if pi0[0] != pj0[0]:
                    w_r += abs(pi1[0]-pj1[0])/abs(pi0[0]-pj0[0])
                if pi0[1] != pj0[1]:
                    h_r += abs(pi1[1]-pj1[1])/abs(pi0[1]-pj0[1])
        self.h_r = 2*h_r/n/(n-1)
        self.w_r = 2*w_r/n/(n-1)
        print self.counter, '- Ratio: ', self.h_r, self.w_r

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
            print 'Rescaling...'
            core1[0] = core1[0]*self.w_r
            core1[1] = core1[1]*self.h_r
        core2 -= core1
        # print core2
        x = int(core2[0])
        y = int(core2[1])
        self.double.append((y, x))
        if self.show:
            y_ = y
            x_ = x
            if y_ < 0:
                y_ = 0
            if x_ < 0:
                x_ = 0
            scale_w = self.ker_w
            scale_h = self.ker_h
            if self.tag:
                print 'Rescaling Done!'
                scale_w = int(scale_w*self.w_r)
                scale_h = int(scale_h*self.h_r)
            self.triplet.append(self.img2[y_:(y+scale_h), x_:(x+scale_w)])

        if self.show:
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
        y = self.double[0][0]
        x = self.double[0][1]
        y_ = y
        x_ = x
        if y_ < 0:
            y_ = 0
        if x_ < 0:
            x_ = 0
        self.ret1 = self.src2[y_:(y + self.ker_h), x_:(x + self.ker_w)]

        y = self.double[1][0]
        x = self.double[1][1]
        y_ = y
        x_ = x
        if y_ < 0:
            y_ = 0
        if x_ < 0:
            x_ = 0
        self.ret2 = self.src3[y_:(y + self.ker_h), x_:(x + self.ker_w)]

    def update(self):
        if self.x + 1 < self.w_n:
            self.x += 1
        else:
            self.y += 1
            self.x = 0
        if self.y >= self.h_n:
            self.iter = 1

    def getNext(self):
        if self.iter:
            print '    The total number of triplet: {}'.format(self.counter)
            return 0
        y_ = self.h_basis + self.stride_h * self.y
        x_ = self.w_basis + self.stride_w * self.x
        self.img1 = self.src1[y_:(y_ + self.ker_h), x_:(x_ + self.ker_w)]
        self.ref = self.img1

        self.update()

        if self.show:
            self.triplet = [self.img1]

        self.double = []

        if self.show:
            cv2.namedWindow("Crop")
            cv2.moveWindow("Crop", 10, 0)
            cv2.imshow('Crop', self.img1)

        self.match()

        if self.show:
            show_in_one(self.triplet)

        if len(self.double) == 2:
            self.getTriplet()
            self.counter += 1
            if self.counter == 8:
                self.show = 1

            if self.show:
                # show_in_one(self.triplet)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return 2
        if self.show:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 1



class generator(data.Dataset):
    def __init__(self, show, tag):
        super(generator, self).__init__()
        self.disType = []
        self.filenames = []
        self.scores = []
        self.loadFile()

        self.tag = 3
        self.triplet = patch(show, tag)

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
                helpt.append(float(tmp[i]))

            for i in xrange(7):
                dis1 = self.disType[i]
                for j in xrange(i + 1, 8):
                    dis2 = self.disType[j]
                    self.filenames.append([name, name + '_' + tmp[3] + '_' + dis1,
                                           name + '_' + tmp[3] + '_' + dis2,
                                           '{}_{}_{}_'.format(name, dis1, dis2)])
                    if helpt[i] > helpt[j]:
                        self.scores.append(np.array([2]))
                    elif helpt[i] == helpt[j]:
                        self.scores.append(np.array([1]))
                    else:
                        self.scores.append(np.array([0]))
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
                              self.filenames[index][3])
            self.tag = 2
            return
        while self.tag:
            self.tag = self.triplet.getNext()
            if self.tag == 2:
                # self.show()
                return
                # ref = ToTensor()(self.triplet.ref)
                # ret1 = ToTensor()(self.triplet.ret1)
                # ret2 = ToTensor()(self.triplet.ret2)
                # target = torch.LongTensor(self.scores[index])
                # return ref, ret1, ret2, target
            elif self.tag == 0:
                self.tag = 3
                out = open(self.txtName, 'a')
                print self.filenames[index][3]
                # print type(self.filenames[index][3])
                print >> out, self.filenames[index][3], 4
                out.close()
                return

    def __len__(self):
        return len(self.filenames)


data = generator(0, 1) # first - show, second - tag
start = time.time()
for i in xrange(0, 1):
    print '     {}'.format(i)
    s = time.time()
    data[28+26]
    while data.tag == 2:
        data[28+26]
    print '    Step consuming: {}'.format(time.time()-s)
data[0]
print '    Time consuming: {}'.format(time.time()-start)

# data[1]
# while data.tag == 2:
#     data[1]
# data[1]