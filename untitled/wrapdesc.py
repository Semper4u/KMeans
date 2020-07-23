"""
wrapdesc.py是图像的几何变换，仿射变换和透视变换等
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def pic_resize(image):
    """
    放大缩小图片，直接使用resize函数，
    有两种方式，按比例和指定大小，有多种缩放标准，一般使用双内插法
    """
    res = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    h, w = image.shape[:2]
    res2 = cv2.resize(image, (2*h, 1*w), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("img", res)
    cv2.imshow('im', res2)


def pic_loop(image):
    """
    图片平移，是与一个两行三列的矩阵相乘，相乘法则比较特殊，达到x+a,y+b的效果
    注意构造的矩阵类型一定为float
    """
    res = cv2.warpAffine(image, M=np.array([[1, 0, 10], [0, 1, 20]], dtype=float), dsize=image.shape[:2])
    cv2.imshow('loop', res)


def pic_rotate(image):
    """
    图片旋转，是以和平移一样的函数warpAffine,关键点在与构造的矩阵不同
    这里的矩阵重点在与设置旋转中心和旋转角度
    """
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((rows/2, cols/3), -45, 0.8)
    rot = cv2.warpAffine(image, M, dsize=(rows, cols))
    cv2.imshow('rotate', rot)


def pic_affine(image):
    """
    仿射变换在于找到几个点，构造这几个点的仿射变化矩阵
    """
    h, w = image.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[40, 100], [100, 10], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    aff = cv2.warpAffine(image, M, (w*2, h*2))
    # plt.subplot(121, plt.imshow(image), plt.title("input"))
    # plt.subplot(122, plt.imshow(aff), plt.title("output"))
    # plt.show()
    cv2.imshow('output', aff)


def pic_perspective(image):
    """
    构造透视变换，需要找到四个点，任意三个都不能共线，用着四个点构造矩阵
    """
    pts1 = np.float32([[56, 65], [382, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[100, 80], [300, 50], [0, 300], [500, 300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    aff2 = cv2.warpPerspective(image, M, (250, 200))
    # plt.subplot(121, plt.imshow(image), plt.title("input"))
    # plt.subplot(122, plt.imshow(aff), plt.title("output"))
    # plt.show()
    cv2.imshow('perspective', aff2)


if __name__ == '__main__':
    image = cv2.imread("D:/lena.tif")
    print(image.shape)
    while 1:
        cv2.imshow('image', image)
        # pic_resize(image)
        pic_loop(image)
        pic_rotate(image)
        # pic_affine(image)
        pic_perspective(image)
        if cv2.waitKey(1) & 0xff == 27:
            break
    cv2.destroyAllWindows()