"""
contour.py主要是图像的轮廓边界检测与拟合
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_and_draw(image):
    """
    该函数在于查找和绘制轮廓，主要注意调用函数的参数，mode指轮廓检测方式，包含是否检测内部轮廓、拐点等；hierarchy表示轮廓的层次信息
    """
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    ret, thres = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
    cv2.imshow('find_and_draw', img)
    return contours


def draw_fore(image):
    """
    使用查找绘制轮廓可以获得图像的前景信息，方法是先获得图像的轮廓，然后与原图像按位与操作
    """
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    ret, thres = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    cv2.imshow('mask', mask)
    fore = cv2.bitwise_and(image, mask)
    cv2.imshow('fore', fore)


def draw_border(image):
    """
    利用查找绘制轮廓可以更清晰的绘制图像的边缘信息，方法是将canny边缘检测的结果作为绘制的输入，与白色幕布进行绘制
    """
    canny = cv2.Canny(image, 100, 255)
    h = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = h[0]                                # findcontours返回的是一个元祖，第一个为contours
    temp = np.ones(canny.shape, np.uint8)*255           # 创建一个白色幕布
    cv2.drawContours(temp, contours, -1, (0, 255, 0), 1)     # -1表示全画，后面分别是颜色，大小
    cv2.imshow("canny", canny)
    cv2.imshow('contours', temp)


def gain_more(image):
    """
    利用轮廓可以计算每个轮廓的周长和面积信息
    :param contours: 获得的轮廓
    :return: 周长，面积
    """
    contours = find_and_draw(image)
    contours_len = []
    for i in range(len(contours)):
        list_ = [cv2.arcLength(contours[i], True), cv2.contourArea(contours[i])]
        contours_len.append(list_)
        print("第"+str(i)+'个轮廓长度是：%d，面积是：%d' % (contours_len[i][0], contours_len[i][1]))


def epiboly(image):
    """
    该方法创建图像轮廓的最小外包，包括最小外包矩形、圆形、三角形、椭圆和最佳拟合线
    """
    contours = find_and_draw(image)

    # 最小外包矩形
    rect = cv2.minAreaRect(contours[20])
    points = cv2.boxPoints(rect)                               # 找到最小矩形的函数返回值不能直接用于draw，需要转换格式
    points = np.int0(points)
    img = cv2.drawContours(image, [points], 0, (255, 255, 255), 1)
    cv2.imshow('rect', img)

    # 最小外包圆形
    (x, y), radius = cv2.minEnclosingCircle(contours[20])     # 最下圆形，返回圆心和半径
    center = (int(x), int(y))
    rad = int(radius)
    img2 = cv2.circle(image, center, rad, (255, 255, 255), 1)     # 使用cv2.circle画最小圆形
    cv2.imshow('circle', img2)

    # 最小外包三角形
    area, tri = cv2.minEnclosingTriangle(contours[20])       # 最小外包三角形函数返回三角形的面积和三个顶点集
    for i in range(3):
        cv2.line(image, tuple(tri[i][0]), tuple(tri[(i+1) % 3][0]), (255, 255, 255), 1)
    cv2.imshow('tri', image)                                 # 使用line函数画三角形

    # 最小外包椭圆
    ellipse = cv2.fitEllipse(contours[20])
    cv2.ellipse(image, ellipse, (255, 255, 0), 1)
    cv2.imshow('circle', image)

    # 最佳拟合直线
    row, col = image.shape
    [vx, vy, x, y] = cv2.fitLine(contours[20], cv2.DIST_L2, 0, 0.01, 0.01)
    ly = int((-x*vy/vx)+y)
    ry = int(((col-x)*vy/vx)+y)
    cv2.line(image, (col-1, ry), (0, ly), (255, 0, 255), 2)
    cv2.imshow('line', image)


def houghline(image):
    """
    霍夫变换检测直线，霍夫变换通过将图像映射到极坐标系，拟合线段的参数方程，通过投票获得最终参数
    """
    edges = cv2.Canny(image, 100, 255)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=1, minLineLength=15,
                            maxLineGap=1)
    print(lines)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1,), (x2, y2), (255, 255, 255), 2)
    cv2.imshow('houghline', image)


def houghcircle(image):
    """
    霍夫圆检测，霍夫能检测任何能提供参数方程的图形
    """
    edges = cv2.medianBlur(image, 5)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50,
                               param2=20, minRadius=5, maxRadius=25)
    print(circles)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 2)
    cv2.imshow("houghcircle", image)


if __name__ == '__main__':
    image = cv2.imread('D:/lena.tif')
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('original', image)

    find_and_draw(image2)
    draw_fore(image2)
    draw_border(image2)
    gain_more(image2)
    epiboly(image2)
    houghline(image2)
    houghcircle(image2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
