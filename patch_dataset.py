import torch.utils.data as data
from torchvision.transforms import ToTensor
import numpy as np
import torch
import time
import cv2, random
import scipy.io as scio


def readImg(filename):
    """"
    Color image loaded by OpenCV is in BGR mode, but Matplotlib displays in RGB mode.
    cv2.imread(path, style)
        1 - cv2.IMREAD_COLOR
        0 - cv2.IMREAD_GRAYSCALE
        -1 - cv2.IMREAD_UNCHANGED
    """
    print filename
    img = cv2.imread(filename, 1)
    return img


def readMat(filename):
    print filename
    data = scio.loadmat(filename)
    data = data['mapstage2']
    # print data
    # print data.shape, '-', data.dtype, data.size
    return data


class patch():
    def __init__(self):
        self.r = 1.0  # the ratio of threshold
        self.ker_n = 96  # the size of sliding window
        self.stride = self.ker_n/2  # the stride of sliding window

        self.imgDir = '/home/lee/Downloads/RetargetMe/WorkPlace/dataset/'
        self.salDir = '/home/lee/Downloads/RetargetMe/WorkPlace/saliency/'
        self.name = ['sc', 'sc']

        self.sift = cv2.xfeatures2d.SIFT_create()

    def setN(self, refname, retname1, retname2):
        self.counter = 0  # counter the number of the triplets
        method = retname1.split('_')
        self.name[0] = method[2]
        method = retname2.split('_')
        self.name[1] = method[2]
        print self.name

        basisDir = self.imgDir + refname + '/'
        self.src1 = readImg(basisDir + refname + '.png')  # queryImage
        self.sal1 = readMat(self.salDir + refname + '.mat')
        self.sal1_basis = np.average(np.average(self.sal1, axis=1)) * self.r
        # print 'Saliency1:', self.sal1_basis

        self.src2 = readImg(basisDir + retname1 + '.png')  # trainImage
        self.sal2 = readMat(self.salDir + retname1 + '.mat')
        self.sal2_basis = np.average(np.average(self.sal2, axis=1)) * self.r
        # print 'Saliency2:', self.sal2_basis

        self.src3 = readImg(basisDir + retname2 + '.png')  # trainImage
        self.draw3 = self.src3.copy()
        self.sal3 = readMat(self.salDir + retname2 + '.mat')
        self.sal3_basis = np.average(np.average(self.sal3, axis=1)) * self.r
        # print 'Saliency3:', self.sal3_basis

        self.sal_basis = self.sal1_basis

        self.h = self.src1.shape[0]  # the height of reference image
        self.w = self.src1.shape[1]  # the width of reference image
        # print self.src1.shape

        self.scal_h = self.ker_n  # expanding height
        self.scal_w = self.ker_n  # expanding width

        self.h_c = self.src2.shape[0]
        self.w_c = self.src2.shape[1]
        # print self.src2.shape

        self.deltaH = (self.h - self.h_c) / 2  # height basis for crop
        self.deltaW = (self.w - self.w_c) / 2  # width basis for crop
        # print self.deltaH, self.deltaW

        self.h_r = self.h_c * 1.0 / self.h  # the height of retargeted image
        self.w_r = self.w_c * 1.0 / self.w  # the width of retargeted image

        self.h_basis = (self.h % self.stride) / 2
        self.h_n = self.h / self.stride - 1
        self.w_basis = (self.w % self.stride) / 2
        self.w_n = self.w / self.stride - 1

        self.x = 0
        self.y = 0
        self.iter = 0

    def getXY(self, y_, x_):
        x_l = int(x_ * self.w_r - self.scal_w)
        if x_l < 0:
            x_l = 0
        x_r = int(x_ * self.w_r + self.ker_n + self.scal_w)

        y_l = int(y_ * self.h_r - self.scal_h)
        if y_l < 0:
            y_l = 0
        y_r = int(y_ * self.h_r + self.ker_n + self.scal_h)

        if self.name[0] != 'cr':
            pos1 = [y_l, y_r, x_l, x_r]
        else:
            h_l = y_ - self.deltaH - 70
            if h_l < 0:
                h_l = 0
            h_r = y_ + self.ker_n + 70
            w_l = x_ - self.deltaW - 70
            if w_l < 0:
                w_l = 0
            w_r = x_ + self.ker_n + 70
            pos1 = [h_l, h_r, w_l, w_r]
        if self.name[1] != 'cr':
            if self.iter == 0:
                pos2 = [y_l, y_r, x_l, x_r]
            else:
                x_l = int(x_ - self.scal_w)
                if x_l < 0:
                    x_l = 0
                x_r = int(x_ + self.ker_n + self.scal_w)

                y_l = int(y_ - self.scal_h)
                if y_l < 0:
                    y_l = 0
                y_r = int(y_ + self.ker_n + self.scal_h)
                pos2 = [y_l, y_r, x_l, x_r]
        else:
            h_l = y_ - self.deltaH - 70
            if h_l < 0:
                h_l = 0
            h_r = y_ + self.ker_n + 70
            w_l = x_ - self.deltaW - 70
            if w_l < 0:
                w_l = 0
            w_r = x_ + self.ker_n + 70
            pos2 = [h_l, h_r, w_l, w_r]
        return pos1, pos2

    def detectKeypoint(self):
        self.kpts1, des1 = self.sift.detectAndCompute(self.img1, None)
        self.kpts2, des2 = self.sift.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher()

        if des1 is not None and des2 is not None:
            self.matches = bf.match(des1, des2)
            self.matches = sorted(self.matches, key=lambda x: x.distance)
        else:
            self.matches = []

    def RANSAC(self, index):
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

        n = len(consensus_matches)
        if n < 3:
            return 0

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
        core2 -= core1
        # print core2
        y = int(core2[0])
        x = int(core2[1])
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        # print x, y
        self.double.append((x, y))

    def generate(self, i):
        self.detectKeypoint()
        self.RANSAC(i)

    def match(self, pos1, pos2):
        if self.iter == 0:
            self.img2 = self.src2[pos1[0]:pos1[1], pos1[2]:pos1[3]]
            self.generate(1)

            self.img2 = self.src3[pos2[0]:pos2[1], pos2[2]:pos2[3]]
            self.generate(2)
        elif self.iter == 1:
            self.img2 = self.src1[pos1[0]:pos1[1], pos1[2]:pos1[3]]
            self.generate(1)

            self.img2 = self.src3[pos2[0]:pos2[1], pos2[2]:pos2[3]]
            self.generate(2)
        else:
            self.img2 = self.src1[pos1[0]:pos1[1], pos1[2]:pos1[3]]
            self.generate(1)

            self.img2 = self.src2[pos2[0]:pos2[1], pos2[2]:pos2[3]]
            self.generate(2)

    def getTriplet(self, pos1, pos2):
        y = pos1[0] + self.double[0][0]
        x = pos1[2] + self.double[0][1]
        if self.iter == 0:
            self.ret1 = self.src2[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.sal2[y:(y + self.ker_n), x:(x + self.ker_n)] = 0
        else:
            self.ref = self.src1[y:(y + self.ker_n), x:(x + self.ker_n)]

        y = pos2[0] + self.double[1][0]
        x = pos2[2] + self.double[1][1]
        if self.iter == 2:
            self.ret1 = self.src2[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.sal2[y:(y + self.ker_n), x:(x + self.ker_n)] = 0
        else:
            self.ret2 = self.src3[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.sal3[y:(y + self.ker_n), x:(x + self.ker_n)] = 0

    def update(self):
        if self.x + 1 < self.w_n:
            self.x += 1
        else:
            self.y += 1
            self.x = 0
        if self.y >= self.h_n:
            print '*********', self.iter, '**********'
            self.x = 0
            self.y = 0
            self.iter += 1
            if self.iter == 1:
                self.sal_basis = self.sal2_basis
            else:
                self.sal_basis = self.sal3_basis
            self.h_r = self.h * 1.0 / self.h_c  # the height of retargeted image
            self.w_r = self.w * 1.0 / self.w_c  # the width of retargeted image
            self.h_n = self.h_c / self.stride - 1
            self.w_n = self.w_c / self.stride - 1
            self.h_basis = (self.h_c % self.stride) / 2
            self.w_basis = (self.w_c % self.stride) / 2

    def getNext(self):
        if self.iter > 2:
            print '    The total number of triplet:', self.counter
            return 0
        y_ = self.h_basis + self.stride * self.y
        x_ = self.w_basis + self.stride * self.x

        if self.iter == 0:
            self.img1 = self.src1[y_:(y_ + self.ker_n), x_:(x_ + self.ker_n)]
            self.ref = self.img1
            sal = self.sal1[y_:(y_ + self.ker_n), x_:(x_ + self.ker_n)]
        elif self.iter == 1:
            self.img1 = self.src2[y_:(y_ + self.ker_n), x_:(x_ + self.ker_n)]
            self.ret1 = self.img1
            sal = self.sal2[y_:(y_ + self.ker_n), x_:(x_ + self.ker_n)]
        else:
            self.img1 = self.src3[y_:(y_ + self.ker_n), x_:(x_ + self.ker_n)]
            self.ret2 = self.img1
            sal = self.sal3[y_:(y_ + self.ker_n), x_:(x_ + self.ker_n)]

        self.update()
        if np.average(np.average(sal, axis=1)) < self.sal_basis:
            return 1

        pos1, pos2 = self.getXY(y_, x_)
        # print x_l, x_r, '-', y_l, y_r, ':', y_r-y_l

        self.double = []

        self.match(pos1, pos2)

        if len(self.double) == 2:
            self.getTriplet(pos1, pos2)
            if self.ref.shape[0] < self.ker_n or self.ref.shape[1] < self.ker_n:
                return 1
            if self.ret1.shape[0] < self.ker_n or self.ret1.shape[1] < self.ker_n:
                return 1
            if self.ret2.shape[0] < self.ker_n or self.ret2.shape[1] < self.ker_n:
                return 1
            self.counter += 1
            return 2
        return 1


class DatasetFromFolder_Order(data.Dataset):
    def __init__(self, outName=""):
        super(DatasetFromFolder_Order, self).__init__()
        self.disType = []
        self.filenames = []
        self.scores = []
        self.loadFile()

        self.lists = []
        for i in xrange(37):
            self.lists.append(i)
        # random.shuffle(self.lists)

        self.tag = 3
        self.triplet = patch()

        # out = open(outName, 'w')
        # print >> out, self.lists
        # out.close()

    def loadFile(self):
        f = open('/home/lee/Downloads/RetargetMe/WorkPlace/GT/subjRef.txt')
        line = f.readline().strip()
        for part in line.split(' '):
            self.disType.append(part)
        for line in f.readlines():
            tmp = line.split(' ')
            name = tmp[2]

            helpt = []
            for i in xrange(4, len(tmp)):
                helpt.append(float(tmp[i]))

            for i in xrange(8):
                dis1 = self.disType[i]
                for j in xrange(i + 1, 8):
                    dis2 = self.disType[j]
                    self.filenames.append([name, name + '_' + tmp[3] + '_' + dis1,
                                           name + '_' + tmp[3] + '_' + dis2])
                    # self.filenames.append([name, name+'_'+tmp[3]+'_'+dis2+'.png',
                    #                        name + '_' + tmp[3] + '_' + dis1 + '.png'])
                    if helpt[i] > helpt[j]:
                        self.scores.append(np.array([2]))
                        # self.scores.append(np.array([0]))
                        # self.scores.append(np.array([[0, 0, 1]]))
                    elif helpt[i] == helpt[j]:
                        self.scores.append(np.array([1]))
                        # self.scores.append(np.array([1]))
                        # self.scores.append(np.array([[0, 1, 0]]))
                    else:
                        self.scores.append(np.array([0]))
                        # self.scores.append(np.array([2]))
                        # self.scores.append(np.array([[1, 0, 0]]))
        f.close()

    def __getitem__(self, index):
        pic = index / 28
        pair = index % 28
        index = self.lists[pic] * 28 + pair

        # print self.filenames[index]
        if self.tag == 3:
            self.triplet.setN(self.filenames[index][0],
                              self.filenames[index][1],
                              self.filenames[index][2])
        while self.tag:
            self.tag = self.triplet.getNext()
            if self.tag == 2:
                print self.triplet.ref.shape, self.triplet.ret1.shape, self.triplet.ret2.shape                # cv2.namedWindow("ref")
                # cv2.namedWindow('ref')
                # cv2.moveWindow("ref", 10, 30)
                # cv2.imshow('ref', self.triplet.ref)
                # cv2.namedWindow("ret1")
                # cv2.moveWindow("ret1", 10, 230)
                # cv2.imshow('ret1', self.triplet.ret1)
                # cv2.namedWindow("ret2")
                # cv2.moveWindow("ret2", 10, 430)
                # cv2.imshow('ret2', self.triplet.ret2)
                # cv2.waitKey(30)

                # ref = ToTensor()(self.triplet.ref)
                # ret1 = ToTensor()(self.triplet.ret1)
                # ret2 = ToTensor()(self.triplet.ret2)
                #
                # target = torch.LongTensor(self.scores[index])
                # return ref, ret1, ret2, target
                return
            elif self.tag == 0:
                self.tag = 3
                return

    def __len__(self):
        return len(self.filenames)


data = DatasetFromFolder_Order()
num = 37*28
start = time.time()
for i in xrange(0, num):
    s = time.time()
    data[i]
    while data.tag == 2:
        data[i]
    print '    Step consuming: {}'.format(time.time()-s)

print '    Time consuming: {}'.format(time.time()-start)
# cv2.destroyAllWindows()