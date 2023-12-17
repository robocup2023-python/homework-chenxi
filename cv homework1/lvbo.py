import cv2 as cv
import numpy as np
def GaussianFilter(img):
    h,w,c = img.shape
    K_size=3
    sigma=2
    #padding
    pad=10
    out=np.zeros((h+2*pad,w+2*pad,c))
    out[pad:pad+h,pad:pad+w]=img.copy()
    '''for i in range(w):
        out[pad-1,pad+i]=out[pad,pad+i]
    for j in range(h):
        out[pad+j,pad-1]=out[pad+j,pad]
    out[pad-1,pad-1]=out[pad][pad]
    out[pad-1,w+2*pad-1]=out[pad,w+2*pad-2]
    out[h+2*pad-1,pad-1]=out[h+2*pad-2,pad]
    out[h+2*pad-1,w+2*pad-1]=out[h+2*pad-2,w+2*pad-2]'''
    #滤波核
    K=np.zeros((K_size, K_size))
    for x in range(-pad,-pad+K_size):
        for y in range(-pad,-pad+K_size):
            K[y+pad,x+pad]=np.exp(-(x**2+y**2)/(2*(sigma**2)))
    K/=(sigma*sigma*2*np.pi)
    K/=K.sum()
    #与图片进行卷积
    tmp=out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad+y,pad+x,ci]=np.sum(K*tmp[y:y+K_size,x:x+K_size,ci])
    out = out[pad:pad + h, pad:pad + w].astype(np.uint8)
    return out
if __name__ == "__main__":
    Img=cv.imread('C:\school\opencv study\sample.jpg',1)
    Imgnew=GaussianFilter(Img)
    cv.imshow('img',Imgnew)
    if cv.waitKey(0)==27:
        cv.destroyAllWindows()