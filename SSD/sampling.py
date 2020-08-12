import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np

from bbox import bboxIOU

__all__ = ['buildPredBoxes', 'sampleEzDetect']

np.set_printoptions(threshold=np.inf, suppress=True)
torch.set_printoptions(precision=20, threshold=True, linewidth=True)


def buildPredBoxes(config):
    predBoxes = []

    for i in range(len(config.mboxes)):
        l = config.mboxes[i][0]
        wid = config.featureSize[l][0]
        hei = config.featureSize[l][1]

        wbox = config.mboxes[i][1]
        hbox = config.mboxes[i][2]

        for y in range(hei):
            for x in range(wid):
                xc = (x + 0.5) / wid    # x, y每个位置取每个feature map 的中心点来计算
                yc = (y + 0.5) / hei

                '''
                xmin = max(0, xc-wbox/2)
                ymin = max(0, yc-hbox/2)
                xmax = min(0, xc+wbox/2)
                ymax = min(0, yc+hbox/2)
                '''

                xmin = xc-wbox/2
                ymin = yc-hbox/2
                xmax = xc+wbox/2
                ymax = yc+hbox/2

                predBoxes.append([xmin, ymin, xmax, ymax])

    return predBoxes


def sampleEzDetect(config, bboxes):
    '''
    先验框匹配，将根据尺度和长款比生成的predBoxes与真实的目标框进行匹配。匹配遵循两个原则：1.对于每个gt，找到与其IOU值最大的先验框与其匹配，可以保证每个gt至少有一个先验框来预测
    2.对于剩下未匹配的先验框pred，找到与其IOU值大于某个阈值的的gt（这里设置为0.5），与之匹配；
    通过这两个步骤，可以得知，一个gt可以与多个先验框进行匹配，而某个先验框最多与一个gt进行匹配；有匹配的先验框称之为正样本，没有匹配的为负样本；
    :param config: 网络的匹配信息，包括某些层的feature map，长宽比等信息
    :param bboxes:通过预调的ground truth（真实框）
    :return:用于分类和回归预测的选择框selectsamples；
    '''
    # preparing pred boxes
    predBoxes = config.predBoxes   # 预选框

    # preparing groud truth
    truthBoxes = []
    for i in range(len(bboxes)):
        truthBoxes.append([bboxes[i][1], bboxes[i][2], bboxes[i][3], bboxes[i][4]])         # gt

    # compute iou
    iouMatrix = []
    for i in predBoxes:
        ious = []
        for j in truthBoxes:
            ious.append(bboxIOU(i, j))
        iouMatrix.append(ious)                                # IOU值矩阵，根据gt数目不同，可能为1维， 2维，...

    iouMatrix = torch.FloatTensor(iouMatrix)
    iouMatrix2 = iouMatrix.clone()

    ii = 0
    selectedSamples = torch.FloatTensor(128*1024)

    # positive samples from bi-direction match
    for i in range(len(bboxes)):                       # 原则1，选出每个gt中IOU值最大的先验框匹配
        iouViewer = iouMatrix.view(-1)

        iouValues, iouSequence = torch.max(iouViewer, 0)

        predIndex = iouSequence // len(bboxes)
        bboxIndex = iouSequence % len(bboxes)

        if iouValues > 0.1:
            selectedSamples[ii * 6 + 1] = bboxes[bboxIndex][0]
            selectedSamples[ii * 6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii * 6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii * 6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii * 6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii * 6 + 6] = predIndex
            ii = ii + 1
        else:
            break

        iouMatrix[:, bboxIndex] = -1                       # 通过设置-1减小值，进行循环遍历
        iouMatrix[predIndex, :] = -1
        iouMatrix2[predIndex, :] = -1

    # also samples with high iou                             原则2，对于每个prior框，尽量选择匹配，增加正样本
    for i in range(len(predBoxes)):
        v, _ = iouMatrix2[i].max(0)
        predIndex = i
        bboxIndex = _

        if v > 0.7:                         # iou大于0.7的为正样本
            selectedSamples[ii * 6 + 1] = bboxes[bboxIndex][0]
            selectedSamples[ii * 6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii * 6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii * 6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii * 6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii * 6 + 6] = predIndex
            ii += 1

        elif v > 0.5:
            selectedSamples[ii * 6 + 1] = bboxes[bboxIndex][0] * -1
            selectedSamples[ii * 6 + 2] = bboxes[bboxIndex][1]
            selectedSamples[ii * 6 + 3] = bboxes[bboxIndex][2]
            selectedSamples[ii * 6 + 4] = bboxes[bboxIndex][3]
            selectedSamples[ii * 6 + 5] = bboxes[bboxIndex][4]
            selectedSamples[ii * 6 + 6] = predIndex
            ii += 1

    selectedSamples[0] = ii
    return selectedSamples


if __name__ == '__main__':

    bbox = [[2, 104, 78, 375, 183], [2, 133, 88, 197, 123], [4, 26, 189, 44, 238]]

