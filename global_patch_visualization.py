import torch.utils.data as data
import numpy as np
import os, cv2, random, time


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


def draw_in_one(src, global_patch, sup, method, name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    n = len(global_patch)/3
    blank_size = 28

    h = 9*blank_size
    w = src[0].shape[1] + blank_size*3
    w_max = 0
    ker_h = []
    step = 0
    for i in xrange(n):
        h_max = 0
        tmp_w = 0
        for j in xrange(3):
            tmp_h = global_patch[step].shape[0]
            tmp_w += global_patch[step].shape[1]
            h_max = max(h_max, tmp_h)
            step += 1
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
        elif j == 2:
            cv2.putText(merge_img, '{}'.format(sup[1]), (3, h_start - 3), font, 0.8, (255,255,255), 1,
                        cv2.LINE_AA)
            cv2.putText(merge_img, method[1], (42, h_start - 3), font, 0.8, (255, 255, 255), 1,
                        cv2.LINE_AA)
        h_start = h_end + blank_size

    gap = h / n - int(np.mean(ker_h)+1)
    h_start = blank_size
    step = 0
    for i in xrange(n):
        w_start = src[0].shape[1]+blank_size
        cv2.putText(merge_img, '{}'.format(i), (w_start-24, h_start + 60), font, 0.8, (255,255,255), 1,
                        cv2.LINE_AA)
        for j in xrange(3):
            merge_img[h_start:(h_start+global_patch[step].shape[0]), w_start:(w_start+global_patch[step].shape[1])] = global_patch[step]
            w_start += blank_size+global_patch[step].shape[1]
            step += 1
        h_start += gap+ker_h[i]
    cv2.imwrite(name, merge_img)
    # window_name = 'Overall'
    # cv2.namedWindow(window_name)
    # cv2.imshow(window_name, merge_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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
    def __init__(self, output, tag=0):
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

    def setN(self, refname, retname1, retname2, outname, sup):
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

        self.ker_h = self.h/2
        self.ker_w = self.w/2
        self.stride_h = self.ker_h / 2
        self.stride_w = self.ker_w / 2

        self.h_basis = (self.h % self.ker_h ) / 2
        self.w_basis = (self.w % self.ker_w) / 2

        self.x = 0
        self.y = 0
        self.counter = 0
        self.global_patch = []
        self.position_container = []
        self.position = [[0, 0], [0, 0], [0, 0]]

    def detectKeypoint(self):
        self.kpts1, des1 = self.sift.detectAndCompute(self.img1, None)
        self.kpts2, des2 = self.sift.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher()
        self.matches = bf.match(des1, des2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)

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

        h_r = 0.0
        w_r = 0.0
        num_h = n*(n-1)/2
        num_w = num_h
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
                else:
                    num_w -= 1
                if pi0[1] != pj0[1]:
                    h_r += abs(pi1[1]-pj1[1])/abs(pi0[1]-pj0[1])
                else:
                    num_h -= 1
        h_r = min(h_r/num_h, 1.0)
        w_r = min(w_r/num_w, 1.0)
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

    def getNext(self):
        if self.iter:
            src = [self.draw1, self.draw2, self.draw3]
            draw_in_one(src, self.global_patch, self.sup, self.name,  self.overDir + self.filename+'_over.png')
            print '    The total number of triplet: {}'.format(self.counter)
            return 0
        y_ = self.h_basis + self.stride_h * self.y
        x_ = self.w_basis + self.stride_w * self.x
        self.img1 = self.src1[y_:(y_ + self.ker_h), x_:(x_ + self.ker_w)]
        self.ref = self.img1
        self.position[0] = [y_, y_ + self.ker_h, x_, x_ + self.ker_w]

        cv2.rectangle(self.draw1, (x_, y_), (x_ + self.ker_w, y_ + self.ker_h), self.color[self.counter%3], 1)
        cv2.putText(self.draw1, '{}'.format(self.counter), (x_, y_+21), self.font, 0.8,self.color[self.counter%3]
                    , 1,cv2.LINE_AA)

        self.update()

        self.double = []
        self.scale = []

        self.match()

        if len(self.double) == 2:
            self.getTriplet()
            tmp = []
            for i in xrange(3):
                tmp.append(self.position[i])
            self.position_container.append(tmp)
            name = self.outDir + self.filename
            self.global_patch.append(self.ref)
            self.global_patch.append(self.ret1)
            self.global_patch.append(self.ret2)
            if self.output:
                self.saveImg(name + '_{}_{}.bmp'.format(self.counter, 0), self.ref)
                self.saveImg(name + '_{}_{}.bmp'.format(self.counter, 1), self.ret1)
                self.saveImg(name + '_{}_{}.bmp'.format(self.counter, 2), self.ret2)
            self.counter += 1
            return 2
        return 1

    def saveImg(self, outname, img):
        cv2.imwrite(outname, img)


class generator(data.Dataset):
    def __init__(self, output, tag):
        super(generator, self).__init__()
        self.disType = []
        self.filenames = []
        self.scores = []
        self.loadFile()

        self.tag = 3
        self.triplet = patch(output, tag)

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
                              self.scores[index])
            self.tag = 2
            return
        while self.tag:
            self.tag = self.triplet.getNext()
            if self.tag == 2:
                # self.show()
                return
            elif self.tag == 0:
                self.tag = 3
                out = open(self.txtName, 'a')
                print self.filenames[index][3]
                print >> out, self.filenames[index][3], self.triplet.counter
                out.close()
                return

    def __len__(self):
        return len(self.filenames)


# data = generator(output=0, tag=1)
# start = time.time()
# for i in xrange(0, 1):
#     print '     {}'.format(i)
#     s = time.time()
#     data[28+26]
#     while data.tag == 2:
#         data[28+26]
#     print '    Step consuming: {}'.format(time.time()-s)
# print '    Time consuming: {}'.format(time.time()-start)

data = generator(output=0, tag=1)
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
