import cv2
import serial 
#import time 
import numpy as np

def main(): 
    '''port = "COM4" #may change. 
    bluetooth = serial.Serial(port, 9600)
    print("Connected")
    bluetooth.flushInput() #kickstart. 
    print("start")'''
    font = cv2.FONT_HERSHEY_SIMPLEX
    a = np.ones((120,320,3), np.uint8)*255
    cv2.putText(a,'w',(75,45), font, 1,(100,100,100),2,cv2.LINE_AA)
    cv2.rectangle(a,(65,15),(105,55),(100,100,100),2)
    cv2.putText(a,'a',(25,95), font, 1,(100,100,100),2,cv2.LINE_AA)
    cv2.rectangle(a,(15,65),(55,105),(100,100,100),2)
    cv2.putText(a,'s',(75,95), font, 1,(100,100,100),2,cv2.LINE_AA)
    cv2.rectangle(a,(65,65),(105,105),(100,100,100),2)
    cv2.putText(a,'d',(125,95), font, 1,(100,100,100),2,cv2.LINE_AA)
    cv2.rectangle(a,(115,65),(155,105),(100,100,100),2)
    cv2.putText(a,'q',(25,45), font, 1,(100,100,100),2,cv2.LINE_AA)
    cv2.rectangle(a,(15,15),(55,55),(100,100,100),2)
    cv2.putText(a,'e',(125,45), font, 1,(100,100,100),2,cv2.LINE_AA)
    cv2.rectangle(a,(115,15),(155,55),(100,100,100),2)
    cv2.putText(a,'ENTER',(175,95), font, 1,(100,100,100),2,cv2.LINE_AA)
    cv2.rectangle(a,(165,65),(305,105),(100,100,100),2)
    stop = False
    while True:
        cv2.imshow("dummy", a)
        key = cv2.waitKeyEx(1)
        
        a = np.ones((120,320,3), np.uint8)*255
        cv2.putText(a,'w',(75,45), font, 1,(100,100,100),2,cv2.LINE_AA)
        cv2.rectangle(a,(65,15),(105,55),(100,100,100),2)
        cv2.putText(a,'a',(25,95), font, 1,(100,100,100),2,cv2.LINE_AA)
        cv2.rectangle(a,(15,65),(55,105),(100,100,100),2)
        cv2.putText(a,'s',(75,95), font, 1,(100,100,100),2,cv2.LINE_AA)
        cv2.rectangle(a,(65,65),(105,105),(100,100,100),2)
        cv2.putText(a,'d',(125,95), font, 1,(100,100,100),2,cv2.LINE_AA)
        cv2.rectangle(a,(115,65),(155,105),(100,100,100),2)
        cv2.putText(a,'q',(25,45), font, 1,(100,100,100),2,cv2.LINE_AA)
        cv2.rectangle(a,(15,15),(55,55),(100,100,100),2)
        cv2.putText(a,'e',(125,45), font, 1,(100,100,100),2,cv2.LINE_AA)
        cv2.rectangle(a,(115,15),(155,55),(100,100,100),2)
        cv2.putText(a,'ENTER',(175,95), font, 1,(100,100,100),2,cv2.LINE_AA)
        cv2.rectangle(a,(165,65),(305,105),(100,100,100),2)
        
        if key != -1:
            i=0
            if key == 119: 
                i='w'
                cv2.rectangle(a,(65,15),(105,55),(100,100,100),-2)
                cv2.putText(a,'w',(75,45), font, 1,(255,255,255),2,cv2.LINE_AA)
            elif key == 115: 
                i='s'
                cv2.rectangle(a,(65,65),(105,105),(100,100,100),-2)
                cv2.putText(a,'s',(75,95), font, 1,(255,255,255),2,cv2.LINE_AA)
            elif key == 97: 
                i='a'
                cv2.rectangle(a,(15,65),(55,105),(100,100,100),-2)
                cv2.putText(a,'a',(25,95), font, 1,(255,255,255),2,cv2.LINE_AA)
            elif key == 100: 
                i='d'
                cv2.rectangle(a,(115,65),(155,105),(100,100,100),-2)
                cv2.putText(a,'d',(125,95), font, 1,(255,255,255),2,cv2.LINE_AA)
            elif key == 113: 
                i='q'
                cv2.rectangle(a,(15,15),(55,55),(100,100,100),-2)
                cv2.putText(a,'q',(25,45), font, 1,(255,255,255),2,cv2.LINE_AA)
            elif key == 101: 
                i='e'
                cv2.rectangle(a,(115,15),(155,55),(100,100,100),-2)
                cv2.putText(a,'e',(125,45), font, 1,(255,255,255),2,cv2.LINE_AA)
            elif key == 27: 
                i='x'
                cv2.rectangle(a,(165,65),(305,105),(100,100,100),-2)
                cv2.putText(a,'ENTER',(175,95), font, 1,(255,255,255),2,cv2.LINE_AA)
                stop = True
            else: 
                pass
            print('key =', key, i)
            #bluetooth.write(str.encode(str(i)))
            if stop: 
                break
        
    cv2.destroyAllWindows()
        
if __name__ == "__main__": 
    main()
