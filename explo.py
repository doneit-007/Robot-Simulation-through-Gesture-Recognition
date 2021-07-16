import gym
import pix_main_arena
import time
import pybullet as p
import pybullet_data
#import cv2.aruco as aruco
import numpy as np
import math
#import cv2.aruco as aruco
import cv2

import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import load_model
import math
import operator
loaded_model = tf.keras.models.load_model('myt2.h5')
m= 100000
n = 12
square=0
circle=0
def arucocorners(img):
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    return corners
def botpos(img):
    corners = arucocorners(img)
    if(len(corners)==0):
        return []
    botx=(corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
    boty=(corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
    botv=complex(corners[0][0][0][0]-corners[0][0][3][0],corners[0][0][0][1]-corners[0][0][3][1])
    botp=[botx,boty,botv]
    return botp
def colmatbtao(img):		    
    lis = []    
    for i in range(n):
            for j  in range(n):
                r,b,g = img[i*100 + 50,j*100+ 50]
                if r<5 and b<5 and g<5:
                    lis.append([i*100 + 50,j*100 + 50])
    return lis
def check(lis,r):
        img = env.camera_feed()
        img=cv2.resize(img,(720,720))
        img =  img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        img  = cv2.resize(img,(n*100,n*100))
        if len(botpos(img))==0:
            return true
        botx,boty,botv=botpos(img)
        for pair in lis:
            x = pair[0]
            y = pair[1]
            dist=math.sqrt((x-botx)*(x-botx)+(y-boty)*(y-boty))
            if dist<=100:
                return false
        return true    
    
def moveforward1():
    cnt=0
    while cnt<=100:
        p.stepSimulation()
        if cnt%25==0:
            img=env.camera_feed()
        cnt+=1
        p.stepSimulation()
        env.move_husky(4.5,4.5,4.5,4.5)
            #cv2.waitKey(5)
def movebackward():
    cnt=0
    while cnt<=100:
        p.stepSimulation()
        if cnt%25==0:
            img=env.camera_feed()
        cnt+=1
        p.stepSimulation()
        env.move_husky(-2.,-2.5,-2.5,-2.5)
            #cv2.waitKey(5)
def moveforward():
    cnt=0
    while cnt<=100:
        p.stepSimulation()
        if cnt%35==0:
            img=env.camera_feed()
        cnt+=1
        env.move_husky(8.5,8.5,8.5,8.5)
            #cv2.waitKey(5)

def moveleft():
    cnt=0
    while(cnt<=100):
        if(cnt%35==0):
            img=env.camera_feed()
        p.stepSimulation()
        env.move_husky(-5.9,4.9,-5.9,4.9)
        cnt+=1
def moveright():
    cnt=0
    while(cnt<=105):
        if cnt%35==0:
            img=env.camera_feed()
        p.stepSimulation()
        env.move_husky(4.9,-5.9,4.9,-5.9)
        cnt+=1



if __name__=="__main__":
    env = gym.make("pix_main_arena-v0")
    img = env.camera_feed()
    #img=cv2.resize(img,(720,720))
    print(img.shape)
    r = cv2.selectROI(img)
    img =  img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    print(img.shape)
    img  = cv2.resize(img,(n*100,n*100))
    lis = []
    lis = colmatbtao(img)
    print(lis)
    cnt  = 1
    #img  = cv2.imread(r'E:\NITESH FILE\deep learning\Capture.png')

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
        
        #time.sleep(2)
        # Got this from collect-data.py
        # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (150,150)) 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)





        
        test2 = np.array(test_image,dtype='float32')
        test2 = test2/255
        #test2 = test2/255
        test2 = test2.astype(np.float64)
        #print(test2)
        #test2 = (test2-0.0032502487)/(0.0014753705)
        if cnt%25==0:
                # Threshold of blue in HSV space
             '''hMin = cv2.getTrackbarPos('HMin', 'image')
             sMin = cv2.getTrackbarPos('SMin', 'image')
             vMin = cv2.getTrackbarPos('VMin', 'image')
             hMax = cv2.getTrackbarPos('HMax', 'image')
             sMax = cv2.getTrackbarPos('SMax', 'image')
             vMax = cv2.getTrackbarPos('VMax', 'image')
             lower_blue = np.array([hMin, sMin, vMin])
             upper_blue = np.array([hMax, sMax, vMax])
             mask = cv2.inRange(test2, lower_blue, upper_blue)
             test2 = cv2.bitwise_and(test2, test2, mask = mask)'''
             cv2.imshow("test", test2)
             result = loaded_model.predict(test2.reshape(1, 150,150, 1))
             
             pas_key = {'0': result[0][0],
                       '1': result[0][1],
                       '2': result[0][2],
                       '3': result[0][3],
                       '4': result[0][4],
                       '5': result[0][5],
                       '6': result[0][6],
                       'L': result[0][7],
                       'R': result[0][9],
                       'O': result[0][8]}
             prediction = {'Zero': result[0][0],
                          'One': result[0][1],
                          'Two': result[0][2],
                          'Three': result[0][3],
                          'Four': result[0][4],
                          'Five': result[0][5],
                          'Six': result[0][6],
                          'Left': result[0][7],
                          'Right': result[0][9],
                          'Ok': result[0][8]}
             prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                    #print(prediction)
             pas_key = sorted(pas_key.items(), key=operator.itemgetter(1), reverse=True)
             cv2.putText(frame,pas_key[0][0],(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
             if pas_key[0][0]=='5':
                moveforward()
             if pas_key[0][0]=='2':
                moveright()
             if pas_key[0][0]=='0':
                moveleft()
             if pas_key[0][0]=='3':
                movebackward()
             print(pas_key[0][0])
        cnt = cnt+1
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()







            
