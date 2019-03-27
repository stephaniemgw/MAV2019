import cv2  
import numpy as np 
import time
from itertools import combinations
from itertools import groupby
from operator import itemgetter
import more_itertools as mit
from collections import defaultdict
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
    pixels = list(range(1,201))
    for line in array:
        left_boundary = list(range(int(list(line)[0]-35),int(list(line)[0])))
        pixels = [item for item in pixels if item not in left_boundary]
        right_boundary = list(range(int(list(line)[0]),int(list(line)[0])+35))
        pixels = [item for item in pixels if item not in right_boundary]  
    continuous = [list(group) for group in mit.consecutive_groups(pixels)]
    new_continuous = []
    for sublist in continuous:
        for item in sublist:
            new_continuous.append(item)
    continuous = new_continuous
    ranges = []  
    for k,g in groupby(enumerate(continuous),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        ranges.append((group[0],group[-1]))
    biggest = 0 
    direction = 100
#    print(ranges)
    rangeset_best = 'dummy'
    for rangeset in ranges:
        if (rangeset[1]-rangeset[0])>biggest :
            biggest = (rangeset[1]-rangeset[0])
            rangeset_best = rangeset
    direction = rangeset_best[0] + ((rangeset_best[1]-rangeset_best[0])/2)
    return angle_lst, array, direction
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
cap = cv2.VideoCapture('test5.mp4') 
while True: 
    start_time = time.time()
    ret, frame = cap.read()
    frame = frame[600:,:]
    frame = cv2.resize(frame,(200,100),interpolation=cv2.INTER_AREA)
    bilateral_filtered_image = cv2.bilateralFilter(frame, 5, 175, 175)
    hsv = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(bilateral_filtered_image,100,200)
    lines = cv2.HoughLines(edges,1,np.pi/180,40,min_theta=-0.15,max_theta=0.15)
    lines_vertical = []
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
                lines_vertical.append([[x1,y1],[x2,y2]])
                cv2.line(frame,(x1,y1),(x2,y2),(0,244,255),1)
    if lines is not None:
        lines_vertical = np.unique(lines_vertical, axis=0)
        intersections = []
        lines_vertical = np.unique(lines_vertical, axis=0)
        for line in lines_vertical:
            intersections.append(line_intersection(line,[[-1,100],[220,100]]))
        for intersection in list(intersections):
            if len(intersection)==2 and intersection[0] is not np.nan :    
                cv2.circle(frame,(int(intersection[0]),int(intersection[1])), 5, (0,0,255), -1)
            else:    
                None
    intersections = np.asarray(intersections)
    rows = np.where((intersections[:,0] > 15.0) & (intersections[:,0] < 188.0))
    intersections = intersections[rows]
    direction = 100
    if not len(intersections)==0 :
        angle_lst, new_array,direction = find_objects(intersections)
    cv2.line(frame,(int(direction),-1000),(int(direction),1000),(0,244,255),5)
    cv2.imshow('Edges',frame)
    k = cv2.waitKey(27) & 0xFF
    if k == 27: 
        break 
                 