import cv2
import numpy as np 
from Digits import Digits_Predict_test as Digits_Predict
from random import randint
from scipy import stats
import time 
#import serial

def nothing(x):
    #nothing 
    pass

def contour_check(gray): 
    global best_cnt
    kernel = np.ones((3,3),np.uint8)
    ori = cv2.dilate(gray, kernel)
    gray = cv2.erode(gray, kernel)
    #gray = cv2.dilate(gray, kernel)
    _, contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True) #Sort contours from big to small.
    cnt_results = []
    for cnt in contours: 
        peri = cv2.arcLength(cnt, True)
        approx = cv2.convexHull(cnt) 
        approx = cv2.approxPolyDP(approx, 0.045*peri, True)
        #approx = cv2.approxPolyDP(cnt, 0.025*peri, True)
        area = cv2.contourArea(cnt)
        #cv2.drawContours(img, [approx], -1, (randint(0, 255),randint(0, 255),randint(0, 255)),1)
        if len(approx) is not 4 or area < 1000: 
            continue
        if area < pow(cv2.arcLength(cnt, True)/4, 2) /2: 
            continue
        
        largest_cnt = [approx]
        ax, ay, aw, ah = cv2.boundingRect(largest_cnt[0])
        roi = ori[ay:ay+ah, ax:ax+aw]
        cv2.imshow("roi_pre", roi)
        pts2 = np.float32([[0,0], [int(ah*1.79),0], [0,ah], [int(ah*1.79), ah]])
        pts1 = pts2.copy()
        dis_1 = [] 
        dis_2 = []
        dis_3 = []
        dis_4 = []
        pts_loc = []
        for pt in largest_cnt[0]: 
            pt = pt[0]
            dis_1.append(np.sqrt((pt[0]-ax)**2 + (pt[1]-ay)**2))      #1  2
            dis_2.append(np.sqrt((pt[0]-ax-aw)**2 + (pt[1]-ay)**2))   #3  4
            dis_3.append(np.sqrt((pt[0]-ax)**2 + (pt[1]-ay-ah)**2))
            dis_4.append(np.sqrt((pt[0]-ax-aw)**2 + (pt[1]-ay-ah)**2))
            pts_loc.append([(pt[0]-ax), (pt[1]-ay)])
        pt1 = pts_loc[dis_1.index(min(dis_1))]
        pt2 = pts_loc[dis_2.index(min(dis_2))]
        pt3 = pts_loc[dis_3.index(min(dis_3))]
        pt4 = pts_loc[dis_4.index(min(dis_4))]
        pts1 = np.float32([pt1, pt2, pt3, pt4])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(roi, M, (int(ah*1.79), ah))
        dst = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
        img28 = roi_process(dst)
        h = len(dst)
        w = len(dst[0])
        if h < 28: 
            h = 28
        if w < 28: 
            w = 28
        out = np.zeros((h, w), dtype=float)
        out[0:len(dst), 0:len(dst[0])] = dst
        out[0:28, 0:28] = img28
        
        try: 
            global pred, proba
            start = time.clock()
            pred, proba, g = Digits_Predict.predict(img28)
            end = time.clock() 
            print("ET {}".format(abs(end-start)*1000))
            try: 
                dst[28:56, 0:28] = g
            except: 
                pass
            #print(pred, proba)
            if proba[0][pred[0]] > 2000: 
                
                g = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
                
                cnt_results.append((pred[0], proba[0][pred[0]], g, approx))
                
                #cv2.imshow("cnt", out)
        except Exception as e: 
            print(e)
    
    cnt_results = sorted(cnt_results, key=lambda x:x[0], reverse=True)
    if len(cnt_results) > 0: 
        res = cnt_results[0]
        predict = res[0]
        probab = res[1]
        g = res[2]
        approx = res[3]
        best_cnt = approx
        cv2.drawContours(img, [approx], -1, (randint(0, 255),randint(0, 255),randint(0, 255)),3)
        fr = cv2.imread("Img/emp.png")
        fr[175:203, 60:88] = g
        cv2.putText(fr,"Number: "+str([predict]),(60,100),cv2.FONT_HERSHEY_SIMPLEX, 1,(150,150,50),2,cv2.LINE_AA)
        cv2.putText(fr,"Proba : "+str(probab),(60,150),cv2.FONT_HERSHEY_SIMPLEX, 1,(150,150,50),2,cv2.LINE_AA)
        cv2.imshow("Result", fr)
        #com(predict)
    #else: 
    #    cv2.drawContours(img, [best_cnt], -1, (randint(0, 255),randint(0, 255),randint(0, 255)),3)

def com(val): 
    global ondemand
    global a
    print(ondemand)
    if ondemand == 0: #Native: High speed. 
        bluetooth.write(str.encode(str(val)))
        ondemand = -1 
    elif ondemand == 1: #High accuracy mode. 
        print("a: "+str(len(a)))
        if len(a) >= 15: 
            blah = stats.mode([a], axis=None).mode[0]
            bluetooth.write(str.encode(str(blah)))
            a = [] 
            ondemand = -1 
        else: 
            a.append(val)
            
def read_msg(): 
    global ondemand
    leng = bluetooth.inWaiting()
    if (leng > 0): 
        input_data = bluetooth.readline()
        print(str(leng)+":"+str(input_data.decode()))
        if int(input_data.decode()) != 0 and int(input_data.decode()) != 1: 
            print("Invalid message from Bluetooth. ")
        else: 
            ondemand = int(input_data.decode()) 

def find_rect(): 
    _, contours, _ = cv2.findContours(gaus, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #binary #otsu
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10] #Sort contours from big to small.
            
    largest_cnt = []
            
    for cnt in contours :
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015*peri, True)
        if len(approx) == 4:
            cv2.drawContours(img, [approx], -1, (255,0,255),3)
            largest_cnt.append(approx)
            break
        else:
            cv2.drawContours(img, [approx], -1, (255,100,100),1)
            
    return largest_cnt

def roi_process(gaus): 
    if (gaus is not None): 
        final = np.multiply(np.ones((28, 28), dtype=np.uint8), 255)
        _, contours, _ = cv2.findContours(gaus, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True) #Sort contours from big to small.
        for cnt in contours: 
            ax, ay, aw, ah = cv2.boundingRect(cnt)
            peri = 2 * (aw + ah)
            rct_peri = 2*(len(gaus[0]) + len(gaus[:][0]))
            if abs(peri-rct_peri) > rct_peri/2 and peri > rct_peri/5:
                #cv2.drawContours(gaus, [cnt], -1, (100,100,100),3)
                #cv2.rectangle(gaus, (int(ax), int(ay)), (int(ax+aw), int(ay+ah)), (100,100,100),3)
                roi = gaus[int(ay):int(ay+ah),int(ax):int(ax+aw)]
                if ah>aw: 
                    ratio = ah / 26
                    roi = cv2.resize(roi, (int(aw/ratio*1.1), int(ah/ratio)))
                    gap = int((28-int(aw/ratio*1.1))/2)
                    final[1:1+int(ah/ratio), gap:gap+int(aw/ratio*1.1)] = roi
                else: 
                    ratio = aw / 26
                    roi = cv2.resize(roi, (int(aw/ratio), int(ah/ratio*1.1)))
                    gap = int((28-int(ah/ratio*1.1))/2)
                    final[gap:gap+int(ah/ratio*1.1), 1:1+int(aw/ratio)] = roi
                break
        
        return final

def main(): 
    global img 
    global gray 
    global gaus
    global best_cnt
    global bluetooth
    global ondemand
    global a
    
    #port = "COM4" #may change. 
    #bluetooth = serial.Serial(port, 9600)
    #print("Connected")
    #bluetooth.flushInput() #kickstart. 
    #ondemand = -1 
    #a = []
    
    best_cnt = None
    
    #Digits_Predict.main()
    #Digits_Predict.main()
    
    # # #
    #iinndd = 4209
    # # # 
    
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -8.0)
    
    cv2.namedWindow("display")
    cv2.createTrackbar("cn_t1", "display", 0, 300, nothing)
    cv2.createTrackbar("cn_t2", "display", 300, 300, nothing)
    cv2.createTrackbar("thres", "display", 85, 300, nothing)
    cv2.createTrackbar("min_l", "display", 40, 300, nothing)
    cv2.createTrackbar("max_l", "display", 18, 300, nothing)

    cv2.createTrackbar("blk_size", "display", 75, 300, nothing)
    cv2.createTrackbar("c_const", "display", 10, 100, nothing)
    
    while 1:
        try: 
            #read_msg() 
            _, img = cap.read()
            #img = cv2.imread("Test2/gray.png") #Test1/0_1.png
            
            kernel = np.ones((3,3),np.uint8)
            img = cv2.bilateralFilter(img, 11, 17, 17)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            gray = cv2.erode(gray, kernel)
            
            cv2.imshow("gray", gray)
            
            #lwr = cv2.getTrackbarPos("bin_thres", "display")
            blk = cv2.getTrackbarPos("blk_size", "display")
            c = cv2.getTrackbarPos("c_const", "display")
            _,binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, c) #cv2.ADAPTIVE_THRESH_MEAN_C
            #_, otsu = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            cn_t1 = cv2.getTrackbarPos("cn_t1", "display")
            cn_t2 = cv2.getTrackbarPos("cn_t2", "display")
            edges = cv2.Canny(gray, cn_t1, cn_t2, L2gradient=True)
            gaus = np.bitwise_not(np.add(np.bitwise_not(gaus), edges))
            
            thres = cv2.getTrackbarPos("thres", "display")
            min_l = cv2.getTrackbarPos("min_l", "display")
            max_l = cv2.getTrackbarPos("max_l", "display")
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, thres, maxLineGap=max_l, minLineLength=min_l)
            
            for line in lines: 
            	x1, y1, x2, y2 = line[0]
            	cv2.line(gaus, (x1, y1), (x2, y2), (0, 0, 0), 2)
                
            cv2.imshow("gaus_pre", gaus)
            #gaus = cv2.erode(gaus, kernel)
            #gaus = cv2.dilate(gaus, kernel)
            #otsu = cv2.erode(otsu, kernel)
            
            #largest_cnt = find_rect()
        
            contour_check(gaus)
                
#            corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10) #(roi, max_cor, quality_lv/1000, min_dist)
#            
#            for corner in corners: 
#                x, y = corner.ravel()
#                cv2.circle(img, (int(x), int(y)), 3, (255,0,0), -1)#(int(x+ax), int(y+ay)), 3, (255, 0, 0), -1)
        except: 
            pass    
        cv2.imshow("img", img)
        #cv2.imshow("binary", binary)
        cv2.imshow("gaus", gaus)
        #cv2.imshow("otsu", otsu)
        
        
        if (cv2.waitKey(1) == 27):
            break; 
    
    cap.release()
    cv2.destroyAllWindows()
    Digits_Predict.session.close()
    print("This Tensorflow session is closed. ")
    
if __name__ == "__main__": 
    main() 