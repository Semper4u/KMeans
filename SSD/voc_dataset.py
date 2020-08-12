from os import listdir
from os.path import join
from random import random
from PIL import Image, ImageDraw
import xml.etree.ElementTree
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from sampling import sampleEzDetect
from model import EzDetectConfig

__all__ = ['vocClassName', 'vocClassID', 'vocDataset']

np.set_printoptions(suppress=True, threshold=np.inf)
# torch.set_printoptions(profile='full')

vocClassName = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]


def getVOCInfo(xmlFile):
    """
    解析xml标注文件
    :param xmlFile: 需要解析的xml文件
    :return: 返回每个xml文件中的object对象，诸如[{'category_id':'person', 'bbox':[22.0, 23.0, 44.0, 33.0]},{...}]列表，列表中每个字典为一个对象
    """
    root = xml.etree.ElementTree.parse(xmlFile).getroot()
    anns = root.findall('object')                          # 找到子根节点中带有object的对象框

    bboxes = []
    for ann in anns:
        name = ann.find('name').text
        newAnn = {}
        newAnn['category_id'] = name
        bbox = ann.find('bndbox')
        newAnn['bbox'] = [-1, -1, -1, -1]                  # 一行四列表示矩形框大小
        newAnn['bbox'][0] = float(bbox.find('xmin').text)  # 对每个位置输入真实的矩形框大小
        newAnn['bbox'][1] = float(bbox.find('ymin').text)
        newAnn['bbox'][2] = float(bbox.find('xmax').text)
        newAnn['bbox'][3] = float(bbox.find('ymax').text)
        bboxes.append(newAnn)

    return bboxes


class vocDataset(data.Dataset):
    def __init__(self, config, isTraining=True):
        super(vocDataset, self).__init__()
        self.config = config
        self.isTraining = isTraining

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transformer = transforms.Compose([transforms.ToTensor(), normalize])

    def __getitem__(self, index):
        item = None
        if self.isTraining:
            item = allTrainingData[index % len(allTrainingData)]
        else:
            item = allTestingData[index % len(allTestingData)]

        img = Image.open(item[0])                        # item[0]为图像数据
        allBboxes = getVOCInfo(item[1])                  # item[1]为通过getVOCInfo解析出真实label的数据

        imgWidth, imgHeight = img.size

        targetWidth = int((random()*0.25 + 0.75) * imgWidth)
        targetHeight = int((random()*0.25 + 0.75) * imgHeight)

        # 对图像进行随机crop， 并保证bbox的大小
        xmin = int(random()*(imgWidth - targetWidth))
        ymin = int(random()*(imgHeight - targetHeight))

        img = img.crop((xmin, ymin, xmin+targetWidth, ymin+targetHeight))
        img = img.resize((self.config.targetWidth, self.config.targetHeight), Image.BILINEAR)
        imgT = self.transformer(img)*256


        # 调整bbox
        bboxes = []
        for i in allBboxes:
            xl = i['bbox'][0] - xmin
            yt = i['bbox'][1] - ymin
            xr = i['bbox'][2] - xmin
            yb = i['bbox'][3] - ymin

            if xl < 0:
                xl = 0
            if xr >= targetWidth:
                xr = targetWidth - 1
            if yt < 0:
                yt = 0
            if yb >= targetHeight:
                yb = targetHeight - 1

            xl = xl/targetWidth
            xr = xr/targetWidth
            yt = yt/targetHeight
            yb = yb/targetHeight

            if xr - xl >= 0.05 and yb - yt >= 0.05:
                bbox = [vocClassID[i['category_id']], xl, yt, xr, yb]

                bboxes.append(bbox)

        if len(bboxes) == 0:
            return self[index+1]

        target = sampleEzDetect(self.config, bboxes)

        '''
        # 对预测图片进行测试
        draw = ImageDraw.Draw(img)
        num = int(target[0])
        for j in range(0, num):
            offset = j * 6
            if target[offset + 1] < 0:
                break
            
            k = int(target[offset + 6])
            trueBox = [ target[offset + 2], target[offset + 3], 
                        target[offset + 4], target[offset + 5]]
            predBox = self.config.predBoxes[k]
            
            draw.rectangle([trueBox[0]*self.config.targetWidth,
                            trueBox[1]*self.config.targetHeight,
                            trueBox[2]*self.config.targetWidth,
                            trueBox[3]*self.config.targetHeight])
            
            draw.rectangle([predBox[0]*self.config.targetWidth,
                            predBox[1]*self.config.targetHeight,
                            predBox[2]*self.config.targetWidth,
                            predBox[3]*self.config.targetHeight], None, 'red')
            
        del draw
        img.save("/temp/{}.jpg".format(index))
        '''

        return imgT, target

    def __len__(self):
        if self.isTraining:
            num = len(allTrainingData) - (len(allTrainingData) % self.config.batchSize)
            return num
        else:
            num = len(allTestingData) - (len(allTestingData) % self.config.batchSize)
            return num


vocClassID = {}
for i in range(len(vocClassName)):
    vocClassID[vocClassName[i]] = i+1


allTrainingData = []
allTestingData = []

allFloder = ['E:/dataset/pascal_voc/VOCdevkit/VOC2007']

for floder in allFloder:
    imagePath = join(floder, 'JPEGImages')
    infoPath = join(floder, 'Annotations')
    index = 0

    for f in listdir(imagePath):
        if f.endswith('.jpg'):
            imageFile = join(imagePath, f)
            infoFile = join(infoPath, f[:-4]+'.xml')
            if index % 10 == 0:
                allTestingData.append((imageFile, infoFile))
            else:
                allTrainingData.append((imageFile, infoFile))
            index += 1


if __name__ == '__main__':
    config = EzDetectConfig()
    data = vocDataset(config)
    img, target = data[33]
    print(target.shape)
    print(target)
