import cv2
import matplotlib.pyplot as plt
import numpy as np

Img=cv2.imread('C:\school\opencv study\sample.jpg',1)
gray=cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
gray=np.float32(gray)
dst=cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
Img[dst>0.01*dst.max()]=[0,255,255]
plt.imshow(Img)
plt.show()