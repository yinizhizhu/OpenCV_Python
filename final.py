import cv2, random
import scipy.io as scio
import numpy as np


def draw_in_one(images, name, show_size=(660, 1320), blank_size=6, window_name='Overview'):
    if len(images) < 3:
        return 0
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    # print images[0].dtype
    merge_img = np.zeros(tuple(shape), images[0].dtype)

    count = 0
    h_start = 0
    for i in range(2):
        w_start = 0
        for j in range(3):
            # print count
            im = images[count]
            h_end = h_start + im.shape[0]
            w_end = w_start + im.shape[1]
            tmp_end = w_start + images[count%3].shape[1]
            # print h_start, '-', h_end, ', ', w_start, '-', w_end
            # print im.shape
            if i:
                merge_img[h_start:h_end, w_start:w_end, 2] = im
                w_start = tmp_end + blank_size
            else:
                merge_img[h_start:h_end, w_start:w_end] = im
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(merge_img, name[j],(w_start,h_start+21), font, 0.8,(255,255,255), 1,cv2.LINE_AA)
                w_start = w_end + blank_size
            count += 1
        h_start = h_end+blank_size

    cv2.namedWindow(window_name)
    # cv2.moveWindow(window_name, 450, 570)
    cv2.imshow(window_name, merge_img)


def show_in_one(images, show_size=(72, 213), blank_size=6, window_name='Triplet'):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("ingnore count %s" % (max_count - count))
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 450, 570)
    cv2.imshow(window_name, merge_img)


def readImg(filename):
    """"
    Color image loaded by OpenCV is in BGR mode, but Matplotlib displays in RGB mode.
    cv2.imread(path, style)
        1 - cv2.IMREAD_COLOR
        0 - cv2.IMREAD_GRAYSCALE
        -1 - cv2.IMREAD_UNCHANGED
    """
    img = cv2.imread(filename, 1)
    return img


def readMat(filename):
    data = scio.loadmat(filename)
    data = data['mapstage2']
    # print data
    # print data.shape, '-', data.dtype, data.size
    return data


class patch():
    def __init__(self, show, index1, index2):
        self.show = show    # 1 - show each step of selection and result of matching
        self.counter = 0    # counter the number of the triplets
        self.r = 1.0        # the ratio of threshold
        self.srcRatio = 0.5 # downsample ratio of the source
        self.salRatio = 0.3 # downsample ratio of salient map
        self.color = [(255,0,0),(0,255,0),(0,0,255)]

        self.sift = cv2.xfeatures2d.SIFT_create()

        self.refname = 'pic/ArtRoom'
        self.retnames = ['_0.75_cr', '_0.75_sv', '_0.75_multiop', '_0.75_sc',
                    '_0.75_scl', '_0.75_sm', '_0.75_sns', '_0.75_warp']
        retname = self.refname + self.retnames[index1]
        print retname

        self.src1 = readImg(self.refname + '.png')    # queryImage
        self.draw1 = self.src1.copy()
        self.sal1 = readMat(self.refname + '.mat')
        self.sal1_basis = np.average(np.average(self.sal1, axis=1)) * self.r
        print 'Saliency1:', self.sal1_basis

        self.src2 = readImg(retname + '.png')   # trainImage
        self.name = ['src', retname[17:]]   # name of retargeting algorithm
        self.draw2 = self.src2.copy()
        self.sal2 = readMat(retname + '.mat')
        self.sal2_basis = np.average(np.average(self.sal2, axis=1)) * self.r
        print 'Saliency2:', self.sal2_basis

        retname = self.refname + self.retnames[index2]
        self.name.append(retname[17:])
        self.src3 = readImg(retname + '.png')   # trainImage
        self.draw3 = self.src3.copy()
        self.sal3 = readMat(retname + '.mat')
        self.sal3_basis = np.average(np.average(self.sal3, axis=1)) * self.r
        print 'Saliency3:', self.sal3_basis

        self.sal_basis = self.sal1_basis
        self.ker_n = 64     # the size of sliding window
        self.stride = 32    # the stride of sliding window

        self.h = self.src1.shape[0] # the height of reference image
        self.w = self.src1.shape[1] # the width of reference image
        print self.src1.shape

        self.scal_h = self.ker_n # expanding height
        self.scal_w = self.ker_n # expanding width

        self.h_c = self.src2.shape[0]
        self.w_c = self.src2.shape[1]
        print self.src2.shape

        self.deltaH = (self.h - self.h_c)/2 # height basis for crop
        self.deltaW = (self.w - self.w_c)/2 # width basis for crop
        # print self.deltaH, self.deltaW

        self.h_r = self.h_c *1.0/self.h # the height of retargeted image
        self.w_r = self.w_c *1.0/self.w # the width of retargeted image

        self.h_basis = (self.h%self.stride)/2
        self.h_n = self.h/self.stride - 1
        self.w_basis = (self.w%self.stride)/2
        self.w_n = self.w/self.stride - 1

        self.x = 0
        self.y = 0
        self.iter = 0

        if self.show == 0:
            self.start()

    def start(self):
        tmp = list()
        tmp.append(cv2.resize(self.draw1,
                          (int(self.draw1.shape[1] * self.srcRatio),
                           int(self.draw1.shape[0] * self.srcRatio)),
                          interpolation=cv2.INTER_CUBIC))

        tmp.append(cv2.resize(self.draw2,
                          (int(self.draw2.shape[1] * self.srcRatio),
                           int(self.draw2.shape[0] * self.srcRatio)),
                          interpolation=cv2.INTER_CUBIC))

        tmp.append(cv2.resize(self.draw3,
                          (int(self.draw3.shape[1] * self.srcRatio),
                           int(self.draw3.shape[0] * self.srcRatio)),
                          interpolation=cv2.INTER_CUBIC))

        tmp.append(cv2.resize(self.sal1*255,
                          (int(self.sal1.shape[1]*self.salRatio),
                           int(self.sal1.shape[0] * self.salRatio)),
                          interpolation=cv2.INTER_CUBIC))

        tmp.append(cv2.resize(self.sal2*255,
                          (int(self.sal2.shape[1]*self.salRatio),
                           int(self.sal2.shape[0] * self.salRatio)),
                          interpolation=cv2.INTER_CUBIC)*255)
        tmp.append(cv2.resize(self.sal3*255,
                          (int(self.sal3.shape[1]*self.salRatio),
                           int(self.sal3.shape[0] * self.salRatio)),
                          interpolation=cv2.INTER_CUBIC)*255)
        draw_in_one(tmp, self.name)
        cv2.waitKey(0)

    def getXY(self, y_, x_):
        x_l = int(x_*self.w_r-self.scal_w)
        if x_l < 0:
            x_l = 0
        x_r = int(x_*self.w_r+self.ker_n+self.scal_w)

        y_l = int(y_*self.h_r-self.scal_h)
        if y_l < 0:
            y_l = 0
        y_r = int(y_*self.h_r+self.ker_n+self.scal_h)

        if self.name[0] != 'cr':
            pos1 = [y_l, y_r, x_l, x_r]
        else:
            h_l = y_- self.deltaH - 70
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
                x_l = int(x_-self.scal_w)
                if x_l < 0:
                    x_l = 0
                x_r = int(x_+self.ker_n+self.scal_w)

                y_l = int(y_-self.scal_h)
                if y_l < 0:
                    y_l = 0
                y_r = int(y_+self.ker_n+self.scal_h)
                pos2 = [y_l, y_r, x_l, x_r]
        else:
            h_l = y_- self.deltaH - 70
            if h_l < 0:
                h_l = 0
            h_r = y_ + self.ker_n + 70
            w_l = x_ - self.deltaW - 70
            if w_l < 0:
                w_l = 0
            w_r = x_ + self.ker_n + 70
            pos2 = [h_l, h_r, w_l, w_r]
        return pos1, pos2

    def match(self, pos1, pos2):
        if self.iter == 0:
            self.img2 = self.src2[pos1[0]:pos1[1], pos1[2]:pos1[3]]
            if self.show:
                cv2.namedWindow("Crop1")
                cv2.moveWindow("Crop1", 10, 180)
                cv2.imshow('Crop1', self.img2)
            self.generate(1)

            self.img2 = self.src3[pos2[0]:pos2[1], pos2[2]:pos2[3]]
            if self.show:
                cv2.namedWindow("Crop2")
                cv2.moveWindow("Crop2", 10, 400)
                cv2.imshow('Crop2', self.img2)
            self.generate(2)
        elif self.iter == 1:
            self.img2 = self.src1[pos1[0]:pos1[1], pos1[2]:pos1[3]]
            if self.show:
                cv2.namedWindow("Crop1")
                cv2.moveWindow("Crop1", 10, 180)
                cv2.imshow('Crop1', self.img2)
            self.generate(1)

            self.img2 = self.src3[pos2[0]:pos2[1], pos2[2]:pos2[3]]
            if self.show:
                cv2.namedWindow("Crop2")
                cv2.moveWindow("Crop2", 10, 400)
                cv2.imshow('Crop2', self.img2)
            self.generate(2)
        else:
            self.img2 = self.src1[pos1[0]:pos1[1], pos1[2]:pos1[3]]
            if self.show:
                cv2.namedWindow("Crop1")
                cv2.moveWindow("Crop1", 10, 180)
                cv2.imshow('Crop1', self.img2)
            self.generate(1)

            self.img2 = self.src2[pos2[0]:pos2[1], pos2[2]:pos2[3]]
            if self.show:
                cv2.namedWindow("Crop2")
                cv2.moveWindow("Crop2", 10, 400)
                cv2.imshow('Crop2', self.img2)
            self.generate(2)

    def getTriplet(self, x_, y_, pos1, pos2):
        if self.iter == 0:
            cv2.rectangle(self.draw1,(x_,y_),(x_+self.ker_n,y_+self.ker_n),self.color[0],2)
        elif self.iter == 1:
            cv2.rectangle(self.draw2,(x_,y_),(x_+self.ker_n,y_+self.ker_n),self.color[1],2)
        else:
            cv2.rectangle(self.draw3,(x_,y_),(x_+self.ker_n,y_+self.ker_n),self.color[2],2)

        y = pos1[0] + self.double[0][0]
        x = pos1[2] + self.double[0][1]
        if self.iter == 0:
            self.sal2[y:(y+self.ker_n), x:(x+self.ker_n)] = 0
            cv2.rectangle(self.draw2,(x,y),(x+self.ker_n,y+self.ker_n),self.color[0],2)
        else:
            self.sal1[y:(y+self.ker_n), x:(x+self.ker_n)] = 0
            if self.iter == 1:
                cv2.rectangle(self.draw1,(x,y),(x+self.ker_n,y+self.ker_n),self.color[1],2)
            else:
                cv2.rectangle(self.draw1,(x,y),(x+self.ker_n,y+self.ker_n),self.color[2],2)

        y = pos2[0] + self.double[1][0]
        x = pos2[2] + self.double[1][1]
        if self.iter == 2:
            self.sal2[y:(y+self.ker_n), x:(x+self.ker_n)] = 0
            cv2.rectangle(self.draw2,(x,y),(x+self.ker_n,y+self.ker_n),self.color[2],2)
        else:
            self.sal3[y:(y+self.ker_n), x:(x+self.ker_n)] = 0
            if self.iter == 1:
                cv2.rectangle(self.draw3,(x,y),(x+self.ker_n,y+self.ker_n),self.color[1],2)
            else:
                cv2.rectangle(self.draw3,(x,y),(x+self.ker_n,y+self.ker_n),self.color[0],2)

        tmp = [cv2.resize(self.draw1,
                          (int(self.draw1.shape[1] * self.srcRatio),
                           int(self.draw1.shape[0] * self.srcRatio)),
                          interpolation=cv2.INTER_CUBIC)]
        tmp.append(cv2.resize(self.draw2,
                          (int(self.draw2.shape[1] * self.srcRatio),
                           int(self.draw2.shape[0] * self.srcRatio)),
                          interpolation=cv2.INTER_CUBIC))
        tmp.append(cv2.resize(self.draw3,
                          (int(self.draw3.shape[1] * self.srcRatio),
                           int(self.draw3.shape[0] * self.srcRatio)),
                          interpolation=cv2.INTER_CUBIC))
        tmp.append(cv2.resize(self.sal1*255,
                          (int(self.sal1.shape[1]*self.salRatio),
                           int(self.sal1.shape[0] * self.salRatio)),
                          interpolation=cv2.INTER_CUBIC))
        tmp.append(cv2.resize(self.sal2*255,
                          (int(self.sal2.shape[1]*self.salRatio),
                           int(self.sal2.shape[0] * self.salRatio)),
                          interpolation=cv2.INTER_CUBIC)*255)
        tmp.append(cv2.resize(self.sal3*255,
                          (int(self.sal3.shape[1]*self.salRatio),
                           int(self.sal3.shape[0] * self.salRatio)),
                          interpolation=cv2.INTER_CUBIC)*255)

        if self.show == 0:
            draw_in_one(tmp, self.name)
            cv2.waitKey(30)
            # cv2.destroyAllWindows()

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
            self.h_basis = (self.h_c%self.stride)/2
            self.w_basis = (self.w_c%self.stride)/2

    def getNext(self):
        if self.iter > 2:
            return 0
        y_ = self.h_basis + self.stride*self.y
        x_ = self.w_basis + self.stride*self.x

        if self.iter == 0:
            self.img1 = self.src1[y_:(y_+self.ker_n), x_:(x_+self.ker_n)]
            sal = self.sal1[y_:(y_+self.ker_n), x_:(x_+self.ker_n)]
        elif self.iter == 1:
            self.img1 = self.src2[y_:(y_+self.ker_n), x_:(x_+self.ker_n)]
            sal = self.sal2[y_:(y_+self.ker_n), x_:(x_+self.ker_n)]
        else:
            self.img1 = self.src3[y_:(y_+self.ker_n), x_:(x_+self.ker_n)]
            sal = self.sal3[y_:(y_+self.ker_n), x_:(x_+self.ker_n)]

        self.update()
        if np.average(np.average(sal, axis=1)) < self.sal_basis:
            return 1

        pos1, pos2 = self.getXY(y_, x_)
        # print x_l, x_r, '-', y_l, y_r, ':', y_r-y_l

        if self.show:
            self.triplet = [self.img1]
        self.double = []

        if self.show:
            cv2.namedWindow("Crop")
            cv2.moveWindow("Crop", 10, 0)
            cv2.imshow('Crop', self.img1)

        self.match(pos1, pos2)

        if self.show:
            show_in_one(self.triplet)

        # print len(self.double)
        if len(self.double) == 2:
            self.counter += 1
            # print self.counter
            self.getTriplet(x_, y_, pos1, pos2)

        # if self.show:
        #     saliency = 'Saliency'
        #     cv2.namedWindow(saliency)
        #     cv2.moveWindow(saliency, 900, 600)
        #     cv2.imshow(saliency, self.sal2*255)

        if self.show:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return 1

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
            cv2.moveWindow(matchName, index*450, 50)
            cv2.imshow(matchName, img3)

        tolerance = 10
        consensus_set = []
        if len(self.matches) < 2:
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

        core1 = np.array([0.0,0.0])
        core2 = np.array([0.0,0.0])
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
        if self.show:
            self.triplet.append(self.img2[x:(x+self.ker_n), y:(y+self.ker_n)])

        if self.show:
            ransacName = 'Ransac{}'.format(index)
            cv2.namedWindow(ransacName)
            cv2.moveWindow(ransacName, index*450, 300)
            cv2.imshow(ransacName, matched_image)

    def generate(self, i):
        self.detectKeypoint()
        self.RANSAC(i)

for i in xrange(1, 7):
    for j in xrange(i+1, 8):
        g = patch(0, i, j)
        k = 0
        while g.getNext():
            # print k
            k+=1

        print 'The number of triplet:', g.counter
        cv2.waitKey(0)
        cv2.destroyAllWindows()