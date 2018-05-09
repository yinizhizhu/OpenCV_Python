import torch.utils.data as data
from PIL import Image


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class check():
    def __init__(self):
        self.disType = 'cr'
        self.firstDir = '/home/lee/Downloads/RetargetMe/WorkPlace/dataset/'
        self.loadFile()

    def loadFile(self):
        f = open('subjRef.txt')
        line = f.readline()
        print line
        index = 0
        for line in f.readlines():
            tmp = line.split(' ')
            name = tmp[2]

            helpt = []
            for i in xrange(4, len(tmp)):
                helpt.append(float(tmp[i]))

            # print index, name, tmp[3], self.disType

            basicDir = self.firstDir + name+'/'
            ref = load_img(basicDir+name+'.png')
            ret = load_img(basicDir+name+'_'+tmp[3]+'_'+self.disType+'.png')
            print index, ref.size[0] - ret.size[0], ref.size[1] - ret.size[1]

            index += 1

        f.close()


data = check()