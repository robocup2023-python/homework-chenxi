import cv2 as cv
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
if __name__ == "__main__":
    Img=cv.imread('C:\school\opencv study\sample.jpg',0)
    y=np.fft.fft2(Img)
    y1=np.fft.fftshift(y)
    Y=np.log(np.abs(y1))
    ya=np.angle(y1)
    #频谱
    plt.subplot(131)
    plt.imshow(Y)
    #相位
    plt.subplot(132)
    plt.imshow(ya)
    #进行混合
    Img2=cv.imread('C:\school\opencv study\sample2.jpg', 0)
    y2=np.fft.fft2(Img)
    y21=np.fft.fftshift(y)
    y31=y21+y1
    y3=np.fft.ifftshift(y31)
    imgnew=np.fft.ifft2(y3)
    Imgnew=np.abs(imgnew)
    plt.subplot(133)
    plt.imshow(Imgnew)
    plt.show()