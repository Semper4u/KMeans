"""
segmente.py是图像分割算法运行实例
author: Shirong Liu
encoding: utf-8
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)


def watershed(image, image2, str):
    """
    分水岭算法模仿山地地形，灰度值高处为山峰，地处为山谷
    image：灰度图   image2：原图，三通道图
    """
    # step 1:进行预处理，首先对灰度图像进行阈值处理，然后开运算，开运算可以消除噪声
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k1 = np.ones((3, 3), dtype=np.uint8)                                           # 确定卷积核大小
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k1, iterations=2)

    cv2.imshow('opening', opening)

    # step 2:获得图像的背景图：背景图为开运算+膨胀，在开运算的基础上再膨胀一次，可以确保背景图中不包含任何的前景内容
    bg = cv2.dilate(opening, k1, iterations=3)

    cv2.imshow('background', bg)

    # step 3：获得图像的前景图：对于容易明确分离的可以直接腐蚀，否则需要距离变换函数来实现目标细化、骨架提取、粘连物体分离；
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, fore = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    fore = np.uint8(fore)

    cv2.imshow('foreground', fore)

    # step 4：获得图像的未知区域（边界）：  背景图-前景图
    unknown = cv2.subtract(bg, fore)

    cv2.imshow('unknown', unknown)

    # step 5：对图像进行标注，该步骤旨在避免过度分割
    ret3, makers = cv2.connectedComponents(fore)

    makers = makers + 1                                   # 前景图中背景为0 ，为了避免分水岭算法将其标注为未知，使其加1
    makers[unknown == 255] = 0                           # 将标准的部分白色换色

    # step 6：最终使用分水岭算法
    img = cv2.watershed(image2, makers)        # 要特别注意，输入图像为8位3通道图像，生成marker的是使用灰度图像
    image2[img == -1] = [0, 255, 0]                      # 标准部分为-1的为边界，在最后图像中着重显示

    plt.imshow(img)
    plt.title('marker')
    plt.show()
    cv2.imshow('watershed', image2)

    form_plt(img, str)
    print(ret, ret2, ret3)

def pyramid(image, str):
    """
    图像金字塔算法是利用上采样和下采样来实现图像的恢复与重建
    不管是上采样还是下采样，只要是大小变了，像素点有丢失，图像会失真
    """
    img0 = image
    # 高斯金字塔下采样
    img1 = cv2.pyrDown(img0)        # 第一次下采样
    img2 = cv2.pyrDown(img1)        # 第二次下采样
    img3 = cv2.pyrDown(img2)        # 第三次下采样
    cv2.imshow('img1',img1)
    # 拉普拉斯金字塔
    I0 = img0 - cv2.pyrUp(img1)     # 第一层金字塔， 每一层金字塔由该层图像减去上一层图像的一个上采样
    I1 = img1 - cv2.pyrUp(img2)     # 第二层金字塔
    I2 = img2 - cv2.pyrUp(img3)     # 第三层金字塔
    # 恢复高精度图像
    M0 = I0 + cv2.pyrUp(img1)       # 恢复的第一层， 金字塔保留的是采样过程中丢失的信息，采用金字塔加上上采样可以恢复图像
    M1 = I1 + cv2.pyrUp(img2)       # 恢复的第一层
    M2 = I2 + cv2.pyrUp(img3)       # 恢复的第一层

    list_title = ['img1', 'img2', 'img3', 'I0', 'I1', 'I2', 'M0', 'M1', 'M2']

    # f, ax = plt.subplots(33)
    # axs = ax.flatten()
    for i in range(9):
        cv2.imshow(list_title[i], eval(list_title[i]))
        # axs[i].set_title(list_title[i])
        # axs[i].imshow(eval(list_title[i]))

    form_plt(I2, str[:4])


def form_plt(image, str):
    """
    画图方法，增加代码重用率
    :param image: 需要作图的图像
    :param str: 图像标签
    :return: 作图--直方图
    """
    plt.hist(image.ravel(), 256)
    plt.title("%s" % str)
    plt.show()


if __name__ == '__main__':
    image = cv2.imread('D:/lena.tif')
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    jpg = cv2.imread('D:/3pic.jpg')
    jpg2 = cv2.cvtColor(jpg, cv2.COLOR_BGR2GRAY)

    jpg2 = cv2.resize(jpg2, tuple([8 * i for i in jpg2.shape]), interpolation=cv2.INTER_LINEAR)
    # ret, jpg = cv2.threshold(jpg, 0, 255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(jpg2.shape)
    cv2.imshow("original", image)
    form_plt(jpg, "original")

    watershed(image2, image, "watershed")       # 两个要同时放大哦
    pyramid(image, "pyramid")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
