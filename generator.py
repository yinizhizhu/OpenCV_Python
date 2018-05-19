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


def readMat(filename):
    # print filename
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

        self.imgDir = 'dataset/'
        self.salDir = 'saliency/'
        self.outDir = 'triplet/'
        if os.path.exists(self.outDir) == False:
            os.mkdir(self.outDir)
        self.outDir_top = 'triplet_top10/'
        if os.path.exists(self.outDir_top) == False:
            os.mkdir(self.outDir_top)
        self.name = ['sc', 'sc']

        self.container = []
        self.saliency = 0.0
        self.filename = ''

        self.sift = cv2.xfeatures2d.SIFT_create()

    def setN(self, refname, retname1, retname2, outname):
        if len(self.container):
            t = sorted(self.container, key=lambda c : c[1], reverse=True)
            end = len(t)
            if end > 10:
                end = 10
            for i in xrange(end):
                subname = []
                # print t[i][0], t[i][1]
                for j in xrange(3):
                    subname.append(['_{}_{}.bmp'.format(t[i][0], j),
                                    '_{}_{}.bmp'.format(i, j)])
                for name in subname:
                    img = readImg(self.outDir + self.filename+name[0])
                    cv2.imwrite(self.outDir_top + self.filename+name[1], img)
            self.container = []

        self.counter = 0  # counter the number of the triplets
        method = retname1.split('_')
        self.name[0] = method[-1]
        method = retname2.split('_')
        self.name[1] = method[-1]
        print self.name, outname
        self.outName = outname

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
        self.sal3 = readMat(self.salDir + retname2 + '.mat')
        self.sal3_basis = np.average(np.average(self.sal3, axis=1)) * self.r
        # print 'Saliency3:', self.sal3_basis

        self.sal_basis = self.sal1_basis

        self.h = self.src1.shape[0]  # the height of reference image
        self.w = self.src1.shape[1]  # the width of reference image
        # print self.src1.shape

        self.h_c = self.src2.shape[0]
        self.w_c = self.src2.shape[1]
        # print self.src2.shape

        self.scal_h = (self.h - self.h_c)/2  # expanding height
        self.scal_h += self.ker_n
        self.scal_w = (self.w - self.w_c)/2  # expanding width
        self.scal_w += self.ker_n

        self.h_r = self.h_c * 1.0 / self.h  # the height of retargeted image
        self.w_r = self.w_c * 1.0 / self.w  # the width of retargeted image

        self.h_basis = (self.h % self.stride) / 2
        self.h_n = self.h / self.stride - 1
        self.w_basis = (self.w % self.stride) / 2
        self.w_n = self.w / self.stride - 1

        self.x = 0
        self.y = 0
        self.iter = 0

    def fixXY(self, y_, x_):
        h_l = y_ - self.scal_h
        if h_l < 0:
            h_l = 0
        h_r = y_ + self.ker_n + self.scal_h
        w_l = x_ - self.scal_w
        if w_l < 0:
            w_l = 0
        w_r = x_ + self.ker_n + self.scal_w
        pos = [h_l, h_r, w_l, w_r]
        return pos

    def getXY(self, y_, x_):
        y = int(y_*self.h_r)
        x = int(x_*self.w_r)
        pos1 = self.fixXY(y, x)
        if self.iter == 0:
            pos2 = self.fixXY(y, x)
        else:
            pos2 = self.fixXY(y_, x_)
        return pos1, pos2

    def detectKeypoint(self):
        # print self.img1.shape, self.img2.shape
        if self.img2.shape[0] < self.img1.shape[0] or self.img2.shape[1] < self.img1.shape[1]:
            self.matches = []
            return
        self.kpts1, des1 = self.sift.detectAndCompute(self.img1, None)
        self.kpts2, des2 = self.sift.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher()

        if des1 is not None and des2 is not None:
            self.matches = bf.match(des1, des2)
            self.matches = sorted(self.matches, key=lambda x: x.distance)
        else:
            self.matches = []

    def RANSAC(self):
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

    def generate(self):
        self.detectKeypoint()
        self.RANSAC()

    def match(self, pos1, pos2):
        if self.iter == 0:
            self.img2 = self.src2[pos1[0]:pos1[1], pos1[2]:pos1[3]]
            self.generate()

            self.img2 = self.src3[pos2[0]:pos2[1], pos2[2]:pos2[3]]
            self.generate()
        elif self.iter == 1:
            self.img2 = self.src1[pos1[0]:pos1[1], pos1[2]:pos1[3]]
            self.generate()

            self.img2 = self.src3[pos2[0]:pos2[1], pos2[2]:pos2[3]]
            self.generate()
        else:
            self.img2 = self.src1[pos1[0]:pos1[1], pos1[2]:pos1[3]]
            self.generate()

            self.img2 = self.src2[pos2[0]:pos2[1], pos2[2]:pos2[3]]
            self.generate()

    def getTriplet(self, pos1, pos2):
        y = pos1[0] + self.double[0][0]
        x = pos1[2] + self.double[0][1]
        # print (y, x),
        if self.iter == 0:
            self.ret1 = self.src2[y:(y + self.ker_n), x:(x + self.ker_n)]
            sal = self.sal2[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.saliency += np.sum(np.sum(sal, axis=1))
            self.sal2[y:(y + self.ker_n), x:(x + self.ker_n)] = 0
        else:
            sal = self.sal1[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.saliency += np.sum(np.sum(sal, axis=1))
            self.ref = self.src1[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.sal1[y:(y + self.ker_n), x:(x + self.ker_n)] = 0

        y = pos2[0] + self.double[1][0]
        x = pos2[2] + self.double[1][1]
        # print (y, x)
        if self.iter == 2:
            self.ret1 = self.src2[y:(y + self.ker_n), x:(x + self.ker_n)]
            sal = self.sal2[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.saliency += np.sum(np.sum(sal, axis=1))
            self.sal2[y:(y + self.ker_n), x:(x + self.ker_n)] = 0
        else:
            self.ret2 = self.src3[y:(y + self.ker_n), x:(x + self.ker_n)]
            sal = self.sal3[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.saliency += np.sum(np.sum(sal, axis=1))
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

        self.saliency = np.sum(np.sum(sal, axis=1))
        pos1, pos2 = self.getXY(y_, x_)
        # print x_l, x_r, '-', y_l, y_r, ':', y_r-y_l

        self.double = []
        self.match(pos1, pos2)

        if len(self.double) == 2:
            # print pos1, pos2
            # print self.double
            # print self.counter, (y_, x_),
            self.getTriplet(pos1, pos2)
            # if self.iter:
            #     cv2.namedWindow('ref')
            #     cv2.imshow('ref', self.ref)
            #
            #     cv2.namedWindow('ret1')
            #     cv2.imshow('ret1', self.ret1)
            #
            #     cv2.namedWindow('ret2')
            #     cv2.imshow('ret2', self.ret2)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            if self.ref.shape[0] < self.ker_n or self.ref.shape[1] < self.ker_n:
                return 1
            if self.ret1.shape[0] < self.ker_n or self.ret1.shape[1] < self.ker_n:
                return 1
            if self.ret2.shape[0] < self.ker_n or self.ret2.shape[1] < self.ker_n:
                return 1
            self.filename = self.outName
            name = self.outDir + self.outName
            self.container.append([self.counter, self.saliency])
            self.saveImg(name + '_{}_{}.bmp'.format(self.counter, 0), self.ref)
            self.saveImg(name + '_{}_{}.bmp'.format(self.counter, 1), self.ret1)
            self.saveImg(name + '_{}_{}.bmp'.format(self.counter, 2), self.ret2)
            self.counter += 1
            return 2
        return 1

    def saveImg(self, outname, img):
        cv2.imwrite(outname, img)


class generator(data.Dataset):
    def __init__(self):
        super(generator, self).__init__()
        self.disType = []
        self.filenames = []
        self.scores = []
        self.loadFile()

        self.tag = 3
        self.triplet = patch()

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
                print >> out, self.filenames[index][3], self.triplet.counter
                out.close()
                return

    def __len__(self):
        return len(self.filenames)


data = generator()
start = time.time()
for i in xrange(0, 37):
    print '     {}'.format(i)
    s = time.time()
    data[i*28]
    while data.tag == 2:
        data[i*28]
    print '    Step consuming: {}'.format(time.time()-s)

data[0]
print '    Time consuming: {}'.format(time.time()-start)

# data[1]
# while data.tag == 2:
#     data[1]
# data[1]