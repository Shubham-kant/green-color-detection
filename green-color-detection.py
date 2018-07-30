import numpy as np
import cv2
lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])
kernalopen=np.ones([5,5])
kernalclose=np.ones([20,20])
cap=cv2.VideoCapture(0)
while (cap.isOpened()):
    ret,img=cap.read()
    
    #img=cv2.resize(img,(3400,2200))
    
    hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
    mask =cv2.inRange(hsv,lowerBound,upperBound)
    maskopen =cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernalopen)
    maskclose =cv2.morphologyEx(maskopen,cv2.MORPH_CLOSE,kernalclose)
    maskfinal=maskclose
    _,conts,h=cv2.findContours(maskfinal.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,conts,-1,(0,0,255),2)
    
    cv2.imshow("img",img)
    cv2.imshow("mask",maskfinal)
    ch=cv2.waitKey(10)
    if(ch==ord("q")):
        break
        
cap.release()
cv2.destroyAllWindows()
