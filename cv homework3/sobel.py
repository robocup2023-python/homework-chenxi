import cv2

Img=cv2.imread('C:\school\opencv study\sample.jpg',cv2.IMREAD_UNCHANGED)
sobelx=cv2.Sobel(Img,cv2.CV_64F,dx=1,dy=0)
sobelx=cv2.convertScaleAbs(sobelx)
sobely=cv2.Sobel(Img,cv2.CV_64F,dx=0, dy=1)
sobely=cv2.convertScaleAbs(sobely)
result=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
cv2.imshow("img", result)
cv2.imwrite("sobelresult.jpg",result)
cv2.waitKey()
cv2.destroyAllWindows()