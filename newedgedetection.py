#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:10:37 2019

@author: erwinlodder
"""

import numpy as np
import cv2
cap = cv2.VideoCapture('testvideo_1.mp4') 
  
  
# loop runs if capturing has been initialized 
while True: 
  
    # reads frames from a camera 
    ret, frame = cap.read() 

    row, col= frame.shape[:2]
    bottom= frame[row-2:row, 0:col]
#    mean= cv2.mean(bottom)[0]
    
    bordersize=10
    border=cv2.copyMakeBorder(frame, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    
#    cv2.imshow('image',im)
#    cv2.imshow('bottom',bottom)
    cv2.imshow('border',border)

#%%

  
  
## loop runs if capturing has been initialized 
#while True: 
#  
#    # reads frames from a camera 
#    ret, frame = cap.read() 
    
    
    

#cv2.imshow('Original Image', raw_image)
#cv2.waitKey()


    bilateral_filtered_image = cv2.bilateralFilter(border, 5, 175, 175)
    #cv2.imshow('Bilateral', bilateral_filtered_image)
    
    
    
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    #cv2.imshow('Edge', edge_detected_image)
    
    
    
    _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 10) & (len(approx) < 30000) & (area > 3000) ):
            contour_list.append(contour)
    
    cv2.drawContours(border, contour_list,  -1, (255,0,0), 2)
    cv2.imshow('Objects Detected',border)
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break

cap.release() 
cv2.destroyAllWindows()
