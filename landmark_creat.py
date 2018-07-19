# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 09:31:26 2018

@author: Administrator
"""
import dlib
import os
import numpy as np
#import sys
import cv2

def get_landmarks(im):
    rects = detector(im,1)
    return [[p.x, p.y] for p in predictor(im, rects[0]).parts()]

path='C:\\Users\\Administrator\\Desktop\\face_recognition\\'
predictor_path = path + 'dlib\\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
def txt_creat(name,msg):
    full_path = path + 'FaceAverage\\test\\' + name + '.txt'
    file = open(full_path,'a+')
    file.write(msg)
    file.close()
    
img_path = path + 'FaceAverage\\test'
for filePath in os.listdir(img_path):
    if filePath.endswith(".jpg"): 
        img = cv2.imread(os.path.join(img_path,filePath),0)
        dets = detector(img,1)    
        for k,d in enumerate(dets):
            shape = predictor(img,d)
        points = get_landmarks(img)
        #print(points)        
        for i in range(len(points)):
            l = points
            ss=l[i]
            x=str(ss[0])
            y=str(ss[1])
            p = x + ' '+ y + '\n'
            #print (p)
            txt_creat(filePath,p)
