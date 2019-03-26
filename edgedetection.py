#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:23:54 2019

@author: erwinlodder
"""

# OpenCV program to perform Edge detection in real time 
# import libraries of python OpenCV  
# where its functionality resides 
import cv2  
  
# np is an alias pointing to numpy library 
import numpy as np 
import time
RHOTHETA = 'rhotheta'
ENDPOINT = 'endpoint'
ERRORVAL = np.nan
from itertools import combinations
from itertools import groupby
from operator import itemgetter
import more_itertools as mit
#%%

from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented
def intersection_1(line1, line2, tolerance=1e-6, subpixel=False):
    """Finds the intersection of two lines in rho, theta form.
    Returns closest integer pixel locations. Returns nan if lines are
    parallel or very close to parallel.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    parallelism = abs(theta1 - theta2)
#    print(parallelism)
    if parallelism < tolerance:  
        return [[ERRORVAL, ERRORVAL]]

    # Ax = b    linear system
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)

    if subpixel:
        return [[x0, y0]]
#    print(x0,y0)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
#    print(x0,y0)
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""
#    lines = lines[:-1]
#    print(lines)
    intersections = []
    
    for i, group in enumerate(lines[:-1]):
#        print(group)
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
#                    print(intersection(line1, line2))
#                    print(len(line1),len(line2))
                    testvar = intersection_1(line1, line2)
                    intersections.append(testvar) 
#                print(intersections)
                    

    return intersections

def calc_angle(array):
    angle_lst = []
    for combination in array:
        
        angle = (combination[1][0]-combination[0][0])/(combination[1][1]-combination[0][1])
        angle_lst.append(1/angle)
    angle_array = np.asarray(angle_lst)
    rows = np.where(abs(angle_array[:]) >10.0)
    angle_array = angle_array[rows]

    array = np.asarray(array)[rows]
    return angle_array, array

def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]
            
def find_objects(array):
    array = np.unique(array, axis=0)
    array = list(combinations(list(array), 2))
    angle_lst, array = calc_angle(array)
    pixels = list(range(1,201))
    for line in array:
        left_boundary = list(range(line[0][0]-20,line[0][0]))
        pixels = [item for item in pixels if item not in left_boundary]
        right_boundary = list(range(line[0][0],line[0][0]+20))
        pixels = [item for item in pixels if item not in right_boundary]
    
    continuous = [list(group) for group in mit.consecutive_groups(pixels)]
    new_continuous = []
    for sublist in continuous:
        for item in sublist:
            new_continuous.append(item)
    continuous = new_continuous
#    print(continuous)
    ranges = []
#    print(list(continuous))
    
    for k,g in groupby(enumerate(continuous),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append((group[0],group[-1]))
#        print(ranges)
#    return angle_lst, array, ranges
    biggest = 0 
    direction = 100
#    print(ranges)
    for rangeset in ranges:
        if (rangeset[1]-rangeset[0])>biggest :
            direction = rangeset[0] + ((rangeset[1]-rangeset[0])/2)
            
        
    return angle_lst, array, direction




#%%
cap = cv2.VideoCapture('test4.mp4') 
  
  

while True: 

#    start_time = time.time()
    ret, frame = cap.read()
    bordersize=5
    frame = frame[600:,:]
    frame = cv2.resize(frame,(200,100),interpolation=cv2.INTER_AREA)
    frame=cv2.copyMakeBorder(frame, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
#    blur = cv2.blur(frame,(10,10))    
#    print("--- %s seconds ---" % (time.time() - start_time))

    bilateral_filtered_image = cv2.bilateralFilter(frame, 5, 175, 175)
    hsv = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(bilateral_filtered_image,100,200)

#horizontal   
 
    lines = cv2.HoughLines(edges,1,np.pi/180,70,min_theta=-2.7,max_theta=2.7)

    strong_lines = np.zeros([10,1,2])
    n2 = 0
    if lines is not None:
        for n1 in range(0,len(lines)):
            for rho,theta in lines[n1]:
                if n1 == 0:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1
                else:
                    if rho < 0:
                       rho*=-1
                       theta-=np.pi
                    closeness_rho = np.isclose(rho,strong_lines[0:n2,0,0],atol = 10)
                    closeness_theta = np.isclose(theta,strong_lines[0:n2,0,1],atol = np.pi/36)
                    closeness = np.all([closeness_rho,closeness_theta],axis=0)
                    if not any(closeness) and n2 < 4:
                        strong_lines[n2] = lines[n1]
                        n2 = n2 + 1
    lines_hor = strong_lines
    lines= strong_lines
#    lines_hor = cv2.HoughLines(edges,1,np.pi/180,100,min_theta=-2.5,max_theta=2.5)
    if lines is not None:
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
    
                cv2.line(frame,(x1,y1),(x2,y2),(0,244,255),1)
                
#vertical
    lines = cv2.HoughLines(edges,1,np.pi/180,40,min_theta=-0.15,max_theta=0.15)
    strong_lines = np.zeros([10,1,2])
    n2 = 0
    if lines is not None:
        for n1 in range(0,len(lines)):
            for rho,theta in lines[n1]:
                if n1 == 0:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1
                else:
                    if rho < 0:
                       rho*=-1
                       theta-=np.pi
                    closeness_rho = np.isclose(rho,strong_lines[0:n2,0,0],atol = 10)
                    closeness_theta = np.isclose(theta,strong_lines[0:n2,0,1],atol = np.pi/36)
                    closeness = np.all([closeness_rho,closeness_theta],axis=0)
                    if not any(closeness) and n2 < 4:
                        strong_lines[n2] = lines[n1]
                        n2 = n2 + 1
    lines = strong_lines
    lines_ver = strong_lines
    
    
#    lines_hor = cv2.HoughLines(edges,1,np.pi/180,100,min_theta=-2.5,max_theta=2.5)
    if lines is not None:
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

        
                cv2.line(frame,(x1,y1),(x2,y2),(0,244,255),1)
                
                
    
    if lines is not None:
        lines = np.concatenate((lines_ver, lines_hor), axis=0)
        segmented = segment_by_angle_kmeans(list(lines))
        intersections = segmented_intersections(segmented)
        for intersection in intersections:
            if len(intersection[0])==2 and intersection[0][0] is not np.nan :    
        #                print(intersection)
                cv2.circle(frame,(intersection[0][0],intersection[0][1]), 5, (0,0,255), -1)
            else:    
        #                print(intersection)
                None
#                cv2.circle(frame,(intersection[0][0][0],intersection[0][0][1]), 5, (0,0,255), -1)
#                print('hierwel?')
    intersections = np.squeeze(np.asarray(intersections), axis=1)
    rows = np.where((intersections[:,0] > 15.0) & (intersections[:,0] < 188.0))
    intersections = intersections[rows]
#    print(intersections)
#    rows = np.where((intersections[:,0] > 15.0) & (intersections[:,0] < 388.0))
#    intersections = intersections[rows]
    direction = 100
    if not len(intersections)==0 :
        angle_lst, new_array,direction = find_objects(intersections)
    cv2.line(frame,(int(direction),-1000),(int(direction),1000),(0,244,255),5)
#    print(direction)
#    condlist = [x>0]
#    np.select(condlist, choicelist)
    cv2.imshow('Edges',frame)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    k = cv2.waitKey(27) & 0xFF
    if k == 27: 
        break 
                 
 


def calc_angle(array):
    angle_lst = []
    for combination in array:
        
        angle = (combination[1][0]-combination[0][0])/(combination[1][1]-combination[0][1])
        angle_lst.append(1/angle)
    angle_array = np.asarray(angle_lst)
    rows = np.where(abs(angle_array[:]) >10.0)
    angle_array = angle_array[rows]

    array = np.asarray(array)[rows]
    return angle_array, array

def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]
            
def find_objects(array):
    array = np.unique(array, axis=0)
    array = list(combinations(list(array), 2))
    angle_lst, array = calc_angle(array)
    pixels = list(range(1,201))
    for line in array:
        left_boundary = list(range(line[0][0]-20,line[0][0]))
        pixels = [item for item in pixels if item not in left_boundary]
        right_boundary = list(range(line[0][0],line[0][0]+20))
        pixels = [item for item in pixels if item not in right_boundary]
    
    continuous = [list(group) for group in mit.consecutive_groups(pixels)]
    new_continuous = []
    for sublist in continuous:
        for item in sublist:
            new_continuous.append(item)
    continuous = new_continuous
#    print(continuous)
    ranges = []
#    print(list(continuous))
    
    for k,g in groupby(enumerate(continuous),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append((group[0],group[-1]))
#        print(ranges)
#    return angle_lst, array, ranges
    biggest = 0 
    direction = 100
#    print(ranges)
    for rangeset in ranges:
        if (rangeset[1]-rangeset[0])>biggest :
            direction = rangeset[0] + ((rangeset[1]-rangeset[0])/2)
            
        
    return angle_lst, array, direction
# converting BGR to HSV  


#%%
  
## define range of red color in HSV 
#lower_red = np.array([30,150,50]) 
#upper_red = np.array([255,255,180]) 
#  
## create a red HSV colour boundary and  
## threshold HSV image 
#mask = cv2.inRange(hsv, lower_red, upper_red) 
#  
## Bitwise-AND mask and original image 
#res = cv2.bitwise_and(frame,frame, mask= mask) 
#  
## Display an original image 
##    cv2.imshow('Original',frame) 
#  
## finds edges in the input image image and 
## marks them in the output map edges 
#
#
#lines_ver = cv2.HoughLines(edges,1,np.pi/180,100,min_theta=-0.1,max_theta=0.1)
#lines_hor = cv2.HoughLines(edges,1,np.pi/180,100,min_theta=-2.5,max_theta=2.5)
#lines = lines_ver.append(lines_hor)
#
#
#strong_lines = np.zeros([4,1,2])
#
#
#lines_ver = cv2.HoughLines(edges,1,np.pi/180,100,min_theta=-0.1,max_theta=0.1)
#
#n2 = 0
#for n1 in range(0,len(lines)):
#    for rho,theta in lines[n1]:
#        if n1 == 0:
#            strong_lines[n2] = lines[n1]
#            n2 = n2 + 1
#        else:
#            if rho < 0:
#               rho*=-1
#               theta-=np.pi
#            closeness_rho = np.isclose(rho,strong_lines[0:n2,0,0],atol = 10)
#            closeness_theta = np.isclose(theta,strong_lines[0:n2,0,1],atol = np.pi/36)
#            closeness = np.all([closeness_rho,closeness_theta],axis=0)
#            if not any(closeness) and n2 < 4:
#                strong_lines[n2] = lines[n1]
#                n2 = n2 + 1
#
#if lines is not None:
#    segmented = segment_by_angle_kmeans(list(lines))
##        segmented = segmented[:][:-1]
##        intersections=[]
##        for i, group in enumerate(lines[:-1]):
##            print()
###        print(group)
##            for next_group in lines[i+1:]:
##                for line1 in group:
##                    for line2 in next_group:
##    #                    print(intersection(line1, line2))
##                        
##                        intersections.append([[intersection(line1, line2)]]) 
#intersections = segmented_intersections(segmented)
##        print(intersections)
##    cv2.circle(img,(447,63), 63, (0,0,255), -1)
#
#for intersection in intersections:
#    if len(intersection[0])==2:    
##                print(intersection)
#        cv2.circle(frame,(intersection[0][0],intersection[0][1]), 5, (0,0,255), -1)
#    else:    
##                print(intersection)
#        cv2.circle(frame,(intersection[0][0][0],intersection[0][0][1]), 5, (0,0,255), -1)
##, min_theta=-0.1,max_theta=0.1)
#if lines is not None:
#    for line in lines:
#        for rho,theta in line:
#            a = np.cos(theta)
#            b = np.sin(theta)
#            x0 = a*rho
#            y0 = b*rho
#            x1 = int(x0 + 1000*(-b))
#            y1 = int(y0 + 1000*(a))
#            x2 = int(x0 - 1000*(-b))
#            y2 = int(y0 - 1000*(a))
#    
#            cv2.line(frame,(x1,y1),(x2,y2),(0,244,255),1)
#  
#
#
#
## Display edges in a frame 
#bilateral_filtered_image = cv2.bilateralFilter(edges, 5, 175, 175)
#cv2.imshow('Edges',frame) 
#  
## Wait for Esc key to stop 
##k = cv2.waitKey(5) & 0xFF
##if k == 27: 
##    break
#  
#  
## Close the window 
#cap.release() 
#  
## De-allocate any associated memory usage 
#cv2.destroyAllWindows()  

##%%
#import cv2  
#  
## np is an alias pointing to numpy library 
#import numpy as np 
#  
#  
## capture frames from a camera 
##cap = cv2.VideoCapture('testvideo_1.mp4') 
#  
#  
#
#filename = 'test.png'
#img = cv2.imread(filename)
#frame=img
#  
## converting BGR to HSV 
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
#  
## define range of red color in HSV 
#lower_red = np.array([30,150,50]) 
#upper_red = np.array([255,255,180]) 
#  
## create a red HSV colour boundary and  
## threshold HSV image 
#mask = cv2.inRange(hsv, lower_red, upper_red) 
#  
## Bitwise-AND mask and original image 
#res = cv2.bitwise_and(frame,frame, mask= mask) 
#  
## Display an original image 
##    cv2.imshow('Original',frame) 
#  
## finds edges in the input image image and 
## marks them in the output map edges 
#edges = cv2.Canny(frame,100,200) 
#  
## Display edges in a frame 
#cv2.imshow('Edges',edges) 
#
#  
#  
## Close the window 
##cap.release() 
#cv2.imwrite('messigray.png',edges)
## De-allocate any associated memory usage 
#cv2.destroyAllWindows()
##%%  
#import cv2
#import numpy as np
#
#filename = 'test.png'
#img = cv2.imread(filename)
#frame=img
#  
## converting BGR to HSV 
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
#  
## define range of red color in HSV 
#lower_red = np.array([30,150,50]) 
#upper_red = np.array([255,255,180]) 
#  
## create a red HSV colour boundary and  
## threshold HSV image 
#mask = cv2.inRange(hsv, lower_red, upper_red) 
#  
## Bitwise-AND mask and original image 
#res = cv2.bitwise_and(frame,frame, mask= mask) 
#  
## Display an original image 
##    cv2.imshow('Original',frame) 
#  
## finds edges in the input image image and 
## marks them in the output map edges 
#edges = cv2.Canny(frame,100,200) 
#
#minLineLength = 300
#maxLineGap = 2
#lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength,maxLineGap)
#for x in range(0, len(lines)):
#    for x1,y1,x2,y2 in lines[x]:
#        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#
#cv2.imshow('hough',img)
#cv2.waitKey(0)
##%%
#
#import cv2
#import numpy as np
# 
#img = cv2.imread("test.png")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(gray, 75, 150)
#
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=250)
# 
#for line in lines:
#    x1, y1, x2, y2 = line[0]
#    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# 
#cv2.imshow("Edges", edges)
#cv2.imshow("Image", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#
#
#
#
##%%
## capture frames from a camera 
#import cv2
#import numpy as np
#
#cap = cv2.VideoCapture('testvideo_1.mp4') 
#  
#  
## loop runs if capturing has been initialized 
#while True: 
#    ret, frame = cap.read() 
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    
##    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
##      
##    # define range of red color in HSV 
##    lower_red = np.array([30,150,50]) 
##    upper_red = np.array([255,255,180]) 
##      
##    # create a red HSV colour boundary and  
##    # threshold HSV image 
##    mask = cv2.inRange(hsv, lower_red, upper_red) 
##  
##    # Bitwise-AND mask and original image 
##    res = cv2.bitwise_and(frame,frame, mask= mask) 
##  
#    # Display an original image 
##    cv2.imshow('Original',frame) 
#  
#    # finds edges in the input image image and 
#    # marks them in the output map edges 
##    edges = cv2.Canny(res,100,200) 
#    
#    
#    
#    
#    
#    edges = cv2.Canny(gray, 100, 200)
#    lines = cv2.HoughLines(edges,2,np.pi/180,100, min_theta=-0.2,max_theta=0.2)
#    if lines is not None:
#        for line in lines[:5]:
#            for rho,theta in line:
#                a = np.cos(theta)
#                b = np.sin(theta)
#                x0 = a*rho
#                y0 = b*rho
#                x1 = int(x0 + 1000*(-b))
#                y1 = int(y0 + 1000*(a))
#                x2 = int(x0 - 1000*(-b))
#                y2 = int(y0 - 1000*(a))
#        
#                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
##    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength=100, maxLineGap=20)
##    if lines is not None:
##        for line in lines:
##            x1, y1, x2, y2 = line[0]
##            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     
#    cv2.imshow("Edges", edges)
#    cv2.imshow("Image", frame)
#    k = cv2.waitKey(5) & 0xFF
#    if k == 27: 
#        break
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#
#
#
#
#
#
#
##%%
#import cv2
#import numpy as np
# 
#cap = cv2.VideoCapture('testvideo_1.mp4') 
# 
#while True:
#    ret, frame = cap.read()
#    if not ret:
#        video = cv2.VideoCapture('tesvideo.mp4')
#        continue
# 
#   # reads frames from a camera 
##    ret, frame = cap.read() 
#  
#    # converting BGR to HSV 
#    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
#      
#    # define range of red color in HSV 
#    lower_red = np.array([30,150,50]) 
#    upper_red = np.array([255,255,180]) 
#      
#    # create a red HSV colour boundary and  
#    # threshold HSV image 
#    mask = cv2.inRange(hsv, lower_red, upper_red) 
#  
#    # Bitwise-AND mask and original image 
#    res = cv2.bitwise_and(frame,frame, mask= mask) 
#  
#    # Display an original image 
##    cv2.imshow('Original',frame) 
#  
#    # finds edges in the input image image and 
#    # marks them in the output map edges 
#    edges = cv2.Canny(frame,100,200) 
#  
# 
#    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
#    if lines is not None:
#        for line in lines:
#            x1, y1, x2, y2 = line[0]
#            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
# 
##    cv2.imshow("frame", frame)
#    cv2.imshow("edges", edges)
# 
#    key = cv2.waitKey(25)
#    if key == 27:
#        break
#video.release()
#cv2.destroyAllWindows()