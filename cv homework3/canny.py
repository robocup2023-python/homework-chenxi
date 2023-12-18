import cv2
import matplotlib.pyplot as plt
import numpy as np
def NMS(img, direction):
    W, H = img.shape
    nms = np.copy(img[1:-1, 1:-1])
    for i in range(1, W - 1):
        for j in range(1, H - 1):
            theta = direction[i, j]
            weight = np.tan(theta)
            if theta > np.pi / 4:
                d1 = [0, 1]
                d2 = [1, 1]
                weight = 1 / weight
            elif theta >= 0:
                d1 = [1, 0]
                d2 = [1, 1]
            elif theta >= - np.pi / 4:
                d1 = [1, 0]
                d2 = [1, -1]
                weight *= -1
            else:
                d1 = [0, -1]
                d2 = [1, -1]
                weight = -1 / weight

            g1 = img[i + d1[0], j + d1[1]]
            g2 = img[i + d2[0], j + d2[1]]
            g3 = img[i - d1[0], j - d1[1]]
            g4 = img[i - d2[0], j - d2[1]]

            grade_count1 = g1 * weight + g2 * (1 - weight)
            grade_count2 = g3 * weight + g4 * (1 - weight)

            if grade_count1 > img[i, j] or grade_count2 > img[i, j]:
                nms[i - 1, j - 1] = 0
    return nms
Img=cv2.imread('C:\school\opencv study\sample.jpg',0)
Img = cv2.GaussianBlur(Img, (3, 3), 1)
sobelx=cv2.Sobel(Img,cv2.CV_64F,dx=1,dy=0)
sobelx=cv2.convertScaleAbs(sobelx)
sobely=cv2.Sobel(Img,cv2.CV_64F,dx=0, dy=1)  # y方向的
sobely=cv2.convertScaleAbs(sobely)
sobel=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
h,w=Img.shape
nms=NMS(Img,sobel)
canny=cv2.Canny(nms, 50, 150)
plt.subplot(131)
plt.imshow(sobel)
canny=cv2.Canny(Img,50,150)
plt.subplot(132)
plt.imshow(nms)
plt.subplot(133)
plt.imshow(canny)
plt.show()