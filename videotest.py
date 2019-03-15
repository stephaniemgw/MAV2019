#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:47:27 2019

@author: erwinlodder
"""

import cv2
import numpy as np
 
cap = cv2.VideoCapture('testvideo_4.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_1.mpeg-4', fourcc, 30, (848,480))

while True:
    _, frame = cap.read()
    fram = frame.astype('uint8')
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
 
    lower_blue = np.array([0, 100, 20])
    upper_blue = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
 
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    for contour in contours:
        area = cv2.contourArea(contour)
 
        if area > 5000:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
 
    out.write(frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
 
    key = cv2.waitKey(1)
    if key == 27:
        break
 
cap.release()
out.release()
cv2.destroyAllWindows()
