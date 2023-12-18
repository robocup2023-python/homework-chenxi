import cv2
import matplotlib.pyplot as plt
import numpy as np

Img=cv2.imread('C:\school\opencv study\sample.jpg',0)
hist,bins=np.histogram(Img.flatten(),256,normed=True)
cdf=hist.cumsum()
cdf=(255.0*cdf)/cdf[-1]
Imhist=np.interp(Img.flatten(),bins[:-1],cdf)
Imhist=Imhist.reshape(Img.shape)
plt.subplot(121)
plt.hist(Img.flatten(),256)
plt.subplot(122)
plt.hist(Imhist.flatten(),256)
plt.show()