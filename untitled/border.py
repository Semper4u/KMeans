"""
border.py主要用于边缘检测和图像梯度
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def sobel(image):
    """
    sobel算子是一种模拟计算梯度的差分算子，也是一种基于一阶导数的边缘检测算子，由于有局部平均的作用，也能很好的消除噪声
    参数dx表示水平方向求导，dy表示垂直方向求导，ddepth表示图像深度
    """
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, -1)
    sobelx = cv2.convertScaleAbs(sobelx)                # 取绝对值(像素为整数)

    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, -1)
    sobely = cv2.convertScaleAbs(sobely)

    sobelxy = cv2.Sobel(image, cv2.CV_64F, 1, 1)
    sobelxy = cv2.convertScaleAbs(sobelxy)

    sobelx_Y = cv2.addWeighted(sobelx, 0.5, sobely, 0.9, 0)
    # ret, sobelx_Y = cv2.threshold(sobelx_Y, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    list_sobel = ['sobelx', 'sobely', 'sobelxy', 'sobelx_Y']
    for item in list_sobel:
        cv2.imshow(item, eval(item))


def scharr(image):
    """
    scharr算子是sobel算子的改进版，结构一样，但精度闭sobel算子好，注意scharr方法闭sobel方法少一个参数
    """
    scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharrx = cv2.convertScaleAbs(scharrx)                # 取绝对值(像素为整数)

    scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    scharry = cv2.convertScaleAbs(scharry)

    # scharrxy = cv2.Scharr(image, cv2.CV_64F, 1, 1)      # scharr算子不能同时对x，y方向进行滤波，此时只能进行sobel
    # scharrxy = cv2.convertScaleAbs(scharrxy)

    scharrx_Y = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
    # ret, sobelx_Y = cv2.threshold(sobelx_Y, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    list_scharr = ['scharrx', 'scharry', 'scharrx_Y']
    for item in list_scharr:
        cv2.imshow(item, eval(item))


def canny(image):
    """
    canny边缘检测算法一般优于其他的方法，原因在于canny具有梯度与双阈值处理，canny步骤如下：
    1. 高斯降噪，噪声对边缘影响比较大，可以在canny前手动进行一次；
    2.利用sobel算子分别获得水平与垂直方向dx与dy，开平方和得到边缘强度，双正切函数得到梯度方向；
    3.NMF非极大值抑制，，遍历像素点，保留同一梯度方向上的最大值
    4.双阈值处理，大小分别处理，中间的按照是否能连接来
    """
    canny = cv2.Canny(image, 100, 255)
    canny2 = cv2.Canny(image, 180, 255)
    cv2.imshow('canny', canny)
    cv2.imshow('canny2', canny2)


def laplac(image):
    """
    拉普拉斯算子是二阶微分算子，具有旋转不变性， 可以满足不同方向上的边缘检测要求
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)
    cv2.imshow('laplac', laplacian)


def loG(image):
    """
    高斯拉普拉斯算子，是在拉普拉斯算子运算前进行一次高斯滤波，去除噪声，但是呢两次卷积复杂度太高，，于是有了替代方案
    """
    gaussian = cv2.GaussianBlur(image, (3, 3), 0)
    lapla = cv2.Laplacian(gaussian, cv2.CV_64F, None, 3)
    loG = cv2.convertScaleAbs(lapla)
    cv2.imshow('loG', loG)


if __name__ == '__main__':
    image = cv2.imread("D:/3pic.jpg")
    image = cv2.resize(image, tuple([8*i for i in image.shape[:2]]), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('original', image)
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = cv2.imread("D:/lena.tif")
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original", img)

    # sobel(img)
    # scharr(img)
    canny(img)
    laplac(img)
    loG(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
