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


def draw_in_one(src, top10, sup1, sup2, method, name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_size = 28
    ker_n = 96
    n = len(top10)/3
    h = max((blank_size+ker_n)*n, 2*blank_size+src[0].shape[0]+src[1].shape[0]+src[2].shape[0])
    w = src[0].shape[1] + (blank_size+ker_n)*3

    # print images[0].dtype
    merge_img = np.zeros((h, w, 3), src[0].dtype)

    h_start = 0
    for j in range(3):
        # print count=
        h_end = h_start + src[j].shape[0]
        merge_img[h_start:h_end, 0:src[j].shape[1]] = src[j]
        if j == 1:
            cv2.putText(merge_img, '{}'.format(sup1), (src[j].shape[1], h_start + 24), font, 1.2, (255,255,255), 1,
                        cv2.LINE_AA)
            cv2.putText(merge_img, method[0], (src[j].shape[1], h_start + 60), font, 1.2, (255,255,255), 1,
                        cv2.LINE_AA)
        elif j == 2:
            cv2.putText(merge_img, '{}'.format(sup2), (src[j].shape[1], h_start + 24), font, 1.2, (255,255,255), 1,
                        cv2.LINE_AA)
            cv2.putText(merge_img, method[1], (src[j].shape[1], h_start + 60), font, 1.2, (255, 255, 255), 1,
                        cv2.LINE_AA)
        h_start = h_end + blank_size

    gap = h / n - ker_n
    h_start = blank_size
    for i in xrange(n):
        w_start = src[0].shape[1]+blank_size
        for j in xrange(3):
            merge_img[h_start:(h_start+ker_n), w_start:(w_start+ker_n)] = top10[i*3+j]
            w_start += blank_size+ker_n
        h_start += gap+ker_n
    cv2.imwrite(name, merge_img)
    # window_name = 'Overall'
    # cv2.namedWindow(window_name)
    # cv2.imshow(window_name, merge_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def show_in_one(images, counter, iter, show_size=(102, 306), blank_size=6, window_name='Triplet'):
    print ' ', counter, iter
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


class patch():
    def __init__(self, show):
        self.show = show # 0 - no showing, 1 - show process, 2 - show top10
        self.r = 1.0  # the ratio of threshold
        self.ker_n = 96  # the size of sliding window
        self.stride = self.ker_n/2  # the stride of sliding window
        self.color = [(255,0,0),(0,255,0),(0,0,255)]

        self.imgDir = 'dataset/'
        self.salDir = 'saliency/'
        self.outDir = 'triplet/'
        if os.path.exists(self.outDir) == False:
            os.mkdir(self.outDir)
        self.outDir_top = 'triplet_top10/'
        if os.path.exists(self.outDir_top) == False:
            os.mkdir(self.outDir_top)
        self.overDir_top = 'triplet_top10_overall/'
        if os.path.exists(self.overDir_top) == False:
            os.mkdir(self.overDir_top)
        self.name = ['sc', 'sc']

        self.container = []
        self.saliency = 0.0
        self.refname = ''
        self.retname1 = ''
        self.retname2 = ''
        self.sup1 = 0
        self.sup2 = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.sift = cv2.xfeatures2d.SIFT_create()

    def setN(self, refname, retname1, retname2, outname, sup):
        if len(self.container):
            t = sorted(self.container, key=lambda c : c[1], reverse=True)
            end = len(t)
            if end > 10:
                end = 10
            basisDir = self.imgDir + self.refname + '/'
            src = [readImg(basisDir + self.refname + '.png')]
            src.append(readImg(basisDir + self.retname1 + '.png'))
            src.append(readImg(basisDir + self.retname2 + '.png'))
            top10 = []
            print self.position_container
            for i in xrange(end):
                name = '{}'.format(i)
                # print t[i][0], t[i][1], self.position_container[t[i][0]]
                for j in xrange(3):
                    [y, x] = self.position_container[t[i][0]][j]
                    cv2.rectangle(src[j], (x, y), (x + self.ker_n, y + self.ker_n), self.color[2], 1)
                    cv2.putText(src[j], name, (x, y+21), self.font, 0.8,self.color[2], 1,cv2.LINE_AA)
                    name1 = '_{}_{}.bmp'.format(t[i][0], j)
                    name2 = '_{}_{}.bmp'.format(i, j)
                    img = readImg(self.outDir + self.outname+name1)
                    top10.append(img)
                    cv2.imwrite(self.outDir_top + self.outname+name2, img)
                    if self.show == 2:
                        crop = src[j][y:(y + self.ker_n), x:(x + self.ker_n)]
                        cv2.namedWindow('img')
                        cv2.moveWindow('img', 0, 0)
                        cv2.imshow('img', src[j])
                        cv2.namedWindow('crop')
                        cv2.moveWindow('crop', 1000, 10)
                        cv2.imshow('crop', crop)
                        cv2.namedWindow('top10')
                        cv2.moveWindow('top10', 1000, 400)
                        cv2.imshow('top10', img)
                        cv2.waitKey(900)
            draw_in_one(src, top10, self.sup1, self.sup2, self.name, self.overDir_top + self.refname+'_over.png')
            cv2.imwrite(self.overDir_top + self.refname+'.png', src[0])
            cv2.imwrite(self.overDir_top + self.retname1+'_{}.png'.format(self.sup1), src[1])
            cv2.imwrite(self.overDir_top + self.retname2+'_{}.png'.format(self.sup2), src[2])
            self.container = []
            if self.show == 2:
                cv2.destroyAllWindows()

        self.counter = 0  # counter the number of the triplets
        method = retname1.split('_')
        self.name[0] = method[-1]
        method = retname2.split('_')
        self.name[1] = method[-1]
        print self.name, outname
        self.refname = refname
        self.retname1 = retname1
        self.retname2 = retname2
        self.outname = outname
        self.sup1 = sup[0]
        self.sup2 = sup[1]

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

        self.scal_h = (self.h - self.h_c)/4  # expanding height
        self.scal_h += self.ker_n
        self.scal_w = (self.w - self.w_c)/4  # expanding width
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

        self.position_container = []
        self.position = [[0, 0], [0, 0], [0, 0]]

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
            self.position[1] = [y, x]
        else:
            sal = self.sal1[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.saliency += np.sum(np.sum(sal, axis=1))
            self.ref = self.src1[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.sal1[y:(y + self.ker_n), x:(x + self.ker_n)] = 0
            self.position[0] = [y, x]

        y = pos2[0] + self.double[1][0]
        x = pos2[2] + self.double[1][1]
        # print (y, x)
        if self.iter == 2:
            self.ret1 = self.src2[y:(y + self.ker_n), x:(x + self.ker_n)]
            sal = self.sal2[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.saliency += np.sum(np.sum(sal, axis=1))
            self.sal2[y:(y + self.ker_n), x:(x + self.ker_n)] = 0
            self.position[1] = [y, x]
        else:
            self.ret2 = self.src3[y:(y + self.ker_n), x:(x + self.ker_n)]
            sal = self.sal3[y:(y + self.ker_n), x:(x + self.ker_n)]
            self.saliency += np.sum(np.sum(sal, axis=1))
            self.sal3[y:(y + self.ker_n), x:(x + self.ker_n)] = 0
            self.position[2] = [y, x]

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
        self.position[self.iter] = [y_, x_]

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
            # print self.counter, (y_, x_)
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
            name = self.outDir + self.outname
            self.container.append([self.counter, self.saliency])
            tmp = []
            for i in xrange(3):
                tmp.append(self.position[i])
            self.position_container.append(tmp)
            self.saveImg(name + '_{}_{}.bmp'.format(self.counter, 0), self.ref)
            self.saveImg(name + '_{}_{}.bmp'.format(self.counter, 1), self.ret1)
            self.saveImg(name + '_{}_{}.bmp'.format(self.counter, 2), self.ret2)

            if self.show == 1:
                cv2.namedWindow('ref')
                cv2.moveWindow('ref', 10, 10)
                cv2.imshow('ref', self.ref)
                cv2.namedWindow('ret1')
                cv2.moveWindow('ret1', 10, 200)
                cv2.imshow('ret1', self.ret1)
                cv2.namedWindow('ret2')
                cv2.moveWindow('ret2', 10, 400)
                cv2.imshow('ret2', self.ret2)

                [y, x] = self.position_container[self.counter][0]
                cv2.namedWindow('corp_ref')
                cv2.moveWindow('corp_ref', 500, 10)
                cv2.imshow('corp_ref', self.src1[y:(y+self.ker_n), x:(x+self.ker_n)])
                [y, x] = self.position_container[self.counter][1]
                cv2.namedWindow('corp_ret1')
                cv2.moveWindow('corp_ret1', 500, 200)
                cv2.imshow('corp_ret1', self.src2[y:(y+self.ker_n), x:(x+self.ker_n)])
                [y, x] = self.position_container[self.counter][2]
                cv2.namedWindow('corp_ret2')
                cv2.moveWindow('corp_ret2', 500, 400)
                cv2.imshow('corp_ret2', self.src3[y:(y+self.ker_n), x:(x+self.ker_n)])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            self.counter += 1
            # if self.counter ==100:
            #     self.show = 1
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
        self.triplet = patch(0)

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
                    # if helpt[i] > helpt[j]:
                    #     self.scores.append(np.array([2]))
                    # elif helpt[i] == helpt[j]:
                    #     self.scores.append(np.array([1]))
                    # else:
                    #     self.scores.append(np.array([0]))
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
                              self.scores[index])
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
    index = random.randint(0,27)
    # index = 0
    data[i*28+index]
    while data.tag == 2:
        data[i*28+index]
    print index
    print '    Step consuming: {}'.format(time.time()-s)

data[0]
print '    Time consuming: {}'.format(time.time()-start)

# data[1]
# while data.tag == 2:
#     data[1]
# data[1]