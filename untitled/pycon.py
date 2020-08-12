import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread("D:/111.jpg", cv2.IMREAD_COLOR)
h, w = image.shape[:2]
print(h, w, image.size)
# opencv当中颜色通道为BGR,matplotlib当中读取的为RGB
b, g, r = cv2.split(image)
ima2 = cv2.merge([r, g, b])
# 用plt展示图片效果，与OpenCV展示效果对比
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(ima2)
plt.show()
cv2.imshow("bgr image", image)
cv2.imshow("rgb image:", ima2)
# 图像混合，调节不同图像的权重
dst = cv2.addWeighted(image, 0.65, ima2, 0.35, 0)
cv2.imshow("dst", dst)
# 将图像转换为灰度图，看灰度图的效果

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist_grey = cv2.calcHist([grey], [0], None, [16], [0, 256])
plt.hist(grey.ravel(), 256, [0, 256])
plt.plot(hist_grey)
plt.xlim([0, 256])
plt.show()
cv2.imshow('hist', hist_grey)
cv2.imshow("grey", grey)

# 特定颜色物体追踪，主观感受接近HSV颜色空间，采用掩模方法，进行按位运算
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
low_black = np.array([0, 43, 46])       # 获取想要捕获的颜色的最大值和最小值
high_black = np.array([20, 255, 255])     # 获取红色轮廓
mask = cv2.inRange(hsv, low_black, high_black)   # 根据阈值构建掩模
res = cv2.bitwise_and(image, image, mask=mask)   # 进行按位与运算
cv2.imshow("mask", mask)
cv2.imshow("res", res)

cv2.waitKey(0)
cv2.destroyAllWindows()