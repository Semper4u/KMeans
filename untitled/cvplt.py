import cv2
import numpy as np
print(cv2.__version__)
img = np.zeros((512, 512, 3), np.uint8)
print(img.shape)
cv2.line(img, (0, 0), (511, 511), color=(127, 55, 127), thickness=1)
cv2.circle(img, (400, 66), 66, (10, 100, 255), 1)
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)

cv2.imshow("oo", img)

cv2.waitKey(0)
cv2.destroyAllWindows()