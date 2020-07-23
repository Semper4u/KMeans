"""
直方图绘制，直方图统计分析
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def hist_normal(image):
    """
    直方图正规化，是将图像拉伸，灰度范围扩充到0-255之间，实现图像对比度的调节；
    """
    img = cv2.normalize(image, image, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    plt.figure(figsize=(10, 6))
    plt.hist(img.ravel(), 256)
    plt.title("hist_normal")
    plt.show()   


def hist_equal(image):
    """
    直方图均衡化是增加灰度级范围，将像素点平铺，从而增强对比度
    """
    cv2.imshow("image2", image)
    plt.figure("equal hist")
    plt.hist(image.ravel(), 256, color="b", label="origin")
    img = cv2.equalizeHist(image)
    cv2.imshow("equal_hist", img)
    plt.hist(img.ravel(), 256, color="r", label="equal")
    plt.legend()
    plt.show()


def clahehist(image):
    """
    自适应直方图均衡化，是对不同区域均采用均衡化，相当于不同的区域均有不同范围的bins
    """
    cv2.imshow("image2", image)
    plt.figure("equal hist")
    plt.hist(image.ravel(), 256, color="b", label="origin")
    che = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    dst = che.apply(image)
    cv2.imshow('clahe', dst)
    plt.hist(dst.ravel(), 256, color="y", label='clahe')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    image = cv2.imread("D:/lena.tif", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("D:/3pic.jpg", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.resize(image2, tuple([8*i for i in image2.shape]), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("image", image)
    hist = cv2.calcHist([image], [0], None, [256], [0, 255])
    plt.plot(hist)
    plt.hist(image.ravel(), 256)
    plt.title("original")
    plt.show()
    # hist_normal(image)
    # hist_equal(image2)
    clahehist(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()