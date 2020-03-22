import cv2
import numpy as np 
from Digits import Digits_Predict_test as Digits_Predict
from random import randint
import serial

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
        cv2.drawContours(img, [cnt], -1, (100,100,0),1)
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
            pred, proba, g = Digits_Predict.predict(img28)
            try: 
                dst[28:56, 0:28] = g
            except: 
                pass
            print(pred, proba)
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
        #bluetooth.write(str.encode(str(predict))); 
    #else: 
    #    cv2.drawContours(img, [best_cnt], -1, (randint(0, 255),randint(0, 255),randint(0, 255)),3)

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
    
    #port = "COM4" #may change. 
    #bluetooth = serial.Serial(port, 9600)
    #print("Connected")
    #bluetooth.flushInput() #kickstart. 
    
    best_cnt = None
    
    cap = cv2.VideoCapture(0) #Cam id. 
    
    cv2.namedWindow("display")
    cv2.createTrackbar("l_h", "display", 0, 255, nothing)
    cv2.createTrackbar("l_s", "display", 0, 255, nothing)
    cv2.createTrackbar("l_v", "display", 0, 255, nothing)
    cv2.createTrackbar("u_h", "display", 255, 255, nothing)
    cv2.createTrackbar("u_s", "display", 255, 255, nothing)
    cv2.createTrackbar("u_v", "display", 100, 255, nothing)
    
    while 1:
        try: 
            _, img = cap.read()
            
            kernel = np.ones((3,3),np.uint8)
            img = cv2.bilateralFilter(img, 11, 17, 17)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            l_h = cv2.getTrackbarPos("l_h", "display")
            l_s = cv2.getTrackbarPos("l_s", "display")
            l_v = cv2.getTrackbarPos("l_v", "display")
            u_h = cv2.getTrackbarPos("u_h", "display")
            u_s = cv2.getTrackbarPos("u_s", "display")
            u_v = cv2.getTrackbarPos("u_v", "display")
            lower_blk = np.array([l_h,l_s,l_v])
            upper_blk = np.array([u_h,u_s,u_v])
            mask = cv2.inRange(hsv, lower_blk, upper_blk)
            mask = cv2.bitwise_not(mask)
            mask = cv2.erode(mask, kernel) 
            cv2.imshow("msk", mask)
            contour_check(mask)
                
#            corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10) #(roi, max_cor, quality_lv/1000, min_dist)
#            
#            for corner in corners: 
#                x, y = corner.ravel()
#                cv2.circle(img, (int(x), int(y)), 3, (255,0,0), -1)#(int(x+ax), int(y+ay)), 3, (255, 0, 0), -1)
        except: 
            pass    
        cv2.imshow("res", img)
        #cv2.imshow("binary", binary)
        #cv2.imshow("gaus", gaus)
        #cv2.imshow("otsu", otsu)
        
        
        if (cv2.waitKey(1) == 27):
            break; 
    
    cap.release()
    cv2.destroyAllWindows()
    Digits_Predict.session.close()
    print("This Tensorflow session is closed. ")
    
if __name__ == "__main__": 
    main() 