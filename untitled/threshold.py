"""
thresholld.py是图像的阈值处理文件，阈值处理一般要求输入文件图像为灰度图
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def thresh(image):
    """
    普通二值化阈值处理,有五种不同的确定阈值的方法
    """
    ret, thres1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    ret, thres2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thres3 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
    ret, thres4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
    ret, thres5 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
    titles = ['binary', 'binary_inv', 'tozero', 'tozero_inv', 'trunc']
    images = [thres1, thres2, thres3, thres4, thres5]
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def adaptive_th(image):
    """
    自适应阈值处理，只有一个返回值，从效果来看，高斯阈值优于均值阈值
    """
    th1 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('mean', th1)
    cv2.imshow('gaussian', th2)


def otsus(image, flag):
    """
    otsu's方法进行阈值处理，依然用threshold函数，只是在阈值处理方式选择中有所不同
    同时对比了有没有高斯滤波器的阈值选择的差异，图片的效果等
    :param image: 输入待阈值处理的图像
    :param flag: 是否进行高斯滤波的标志选项
    :return: 两幅图像的效果图
    """
    if flag == 1:
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        res, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY +
                                 cv2.THRESH_OTSU)
        cv2.imshow('blur: %d' % res, th2)
    else:
        pass
    ret, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('thres%d' % ret, th)


if __name__ == '__main__':
    img = cv2.imread("D:/3pic.jpg", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    while True:
        cv2.imshow('image', img)
        image = cv2.resize(img, (8 * h, 8 * w), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('newImage', image)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.xlabel('hist_grey')
        plt.show()
        cv2.imshow('hist', hist)

        thresh(image)
        adaptive_th(image)
        otsus(image, 1)

        if cv2.waitKey(0) & 0xff == 27:
            break
    cv2.destroyAllWindows()
