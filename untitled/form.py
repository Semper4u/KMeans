"""
form.py 进行图像的形态学操作
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def erode(image, str):
    """
    腐蚀操作：遍历像素点，当结构元全部被包含在其中时，返回1，否则0
    """
    k = np.ones((3, 3), dtype=np.uint8)
    img = cv2.erode(image, k, iterations=3)   # 可以使用自定义结构元，亦可以使用函数获取
    cv2.imshow("erode", img)
    form_plt(img, str)


def dilate(image, str):
    """
    膨胀操作，遍历像素点，全部包含结构元且外界包含时，返回1，否则为0
    :param image: 原图像
    :param str: 原图像标签
    :return: 膨胀后的图像
    """
    k = cv2.getStructuringElement(1, (3, 3))      # 通过函数获取结构元
    img = cv2.dilate(image, k, iterations=3)
    cv2.imshow('dilate', img)
    form_plt(img, str)


def morph(image, str):
    """
    形态学梯度运算，是利用膨胀-腐蚀，可以获取边缘信息
    """
    k = cv2.getStructuringElement(1, (3, 3))
    img = cv2.morphologyEx(image, op=cv2.MORPH_GRADIENT, kernel=k, iterations=3)
    cv2.imshow('morph', img)
    form_plt(img, str)


def open_and_close(image, str):
    """
    开运算和闭运算：均是以腐蚀和膨胀为基础的运算，开始先腐蚀，再将结果膨胀，闭运算则顺序相反
    二者操作均是cv2.morphologyEx函数，只是选择的op不同而已
    开运算：对于较大物体可以平滑边界，对于较小物体则可以在细小处分离物体
    闭运算：可以填充物体内部的细小区域，连接临近物体，比如连接字母等
    """
    k = np.ones((3, 3), dtype=np.uint8)
    opening = cv2.morphologyEx(image, op=cv2.MORPH_OPEN, kernel=k, iterations=10)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, k, iterations=10)
    cv2.imshow('open', opening)
    cv2.imshow('close', closing)
    form_plt(opening, str[0])
    form_plt(closing, str[1])


def black_hat_top(image, str):
    """
    黑帽运算和顶帽运算（礼帽运算）：建立在开闭运算的基础上；
    黑帽运算：用原图像-闭运算结果，可以获得比原图像更暗的边缘部分，或者获得图像内部的小孔
    顶帽运算：用原图像-开运算结果，可以获得比原图像更亮的边缘信息，或者获得噪声信息
    """
    k = np.ones((3, 3), dtype=np.uint8)
    blacking = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, k, iterations=3)
    topping = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, k, iterations=3)
    cv2.imshow('blacking', blacking)
    cv2.imshow('toping', topping)
    form_plt(blacking, str[0])
    form_plt(topping, str[1])


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
    image2 = cv2.imread('D:/3pic.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.resize(image2, tuple([8*i for i in image2.shape]), interpolation=cv2.INTER_LINEAR)
    ret, image2 = cv2.threshold(image2, 0, 255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("original", image)
    form_plt(image2, "original")

    algorithms = [erode, dilate, morph, open_and_close, black_hat_top]
    tags = ['erode', 'dilate', 'morph', ['opening', 'closing'], ['black_hat', 'top_hat']]

    for i in range(5):
        algorithms[i](image, tags[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()