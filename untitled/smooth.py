"""
smooth.py是图像的平滑滤波处理相关技术实现
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def gaussian(image):
    """
    高斯滤波器，卷积核权重分布远近不同，满足高斯分布；
    """
    img = cv2.GaussianBlur(image, (5, 5), 0.3, 0.1, borderType=cv2.ADAPTIVE_THRESH_MEAN_C)
    cv2.imshow('gaussian', img)
    plt.figure("gaussian", figsize=(10, 6))
    plt.hist(img.ravel(), 256, color="r")
    plt.title("gaussian")
    plt.show()


def mean(image):
    """
    均值滤波器，卷积核中权值一样，对周围多少像素取平均值，选取的卷积核越大，图像失真越严重
    """
    img = cv2.blur(image, (3, 3))
    cv2.imshow('mean', img)
    plt.figure("mean", figsize=(10, 6))
    plt.hist(img.ravel(), 256, color="r")
    plt.title("mean")
    plt.show()


def boxfilter(image):
    """
    方框滤波器，在均值滤波的基础上，是否选择归一化，若不归一化，便是所有像素之和
    """
    img = cv2.boxFilter(image, -1, (2, 2), normalize=0)
    cv2.imshow('box', img)
    plt.figure("box", figsize=(10, 6))
    plt.hist(img.ravel(), 256, color="r")
    plt.title("box")
    plt.show()


def median(image):
    """
    中值滤波，选择卷积核奇数点的像素值，排序后选择基于中间的像素值，作为ancher的像素值
    """
    img = cv2.medianBlur(image, 5)
    cv2.imshow('median', img)
    plt.figure("median", figsize=(10, 6))
    plt.hist(img.ravel(), 256, color="r")
    plt.title("median")
    plt.show()


def bilateral(image):
    """
    双边滤波，不仅考虑距离信息，也考虑色差信息，与当前像素点色差较小的点会被赋予较大的权值，反之一致；
    """
    img = cv2.bilateralFilter(image, 5, 100, 100)
    cv2.imshow('bilateral', img)
    plt.figure("bilateral", figsize=(10, 6))
    plt.hist(img.ravel(), 256, color="r")
    plt.title("bilateral")
    plt.show()


def diy2d(image):
    """
    自定义卷积核，可以根据自己需要设置卷积核
    """
    k1 = np.random.randint(1, 4, (2, 2), dtype=int)
    img = cv2.filter2D(image, -1, k1)
    img2 = cv2.Laplacian(image, -1, (3, 3))
    cv2.imshow('diy', img2)
    plt.figure("diy", figsize=(10, 6))
    plt.hist(img.ravel(), 256, color="r")
    plt.title("diy")
    plt.show()


if __name__ == '__main__':
    image = cv2.imread("D:/3pic.jpg", flags=cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("D:/lena.tif")
    image = cv2.resize(image, tuple([8*i for i in image.shape]), interpolation=cv2.INTER_LINEAR)
    plt.figure("original", figsize=(10, 6))
    plt.hist(image2.ravel(), 256, color="b")
    plt.title("original")
    plt.show()
    cv2.imshow('original', image2)

    list_func = [gaussian, mean, boxfilter, median, bilateral, diy2d]
    for item in list_func:
        item(image2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
