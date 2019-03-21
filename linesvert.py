# capture frames from a camera 
import cv2
import numpy as np

cap = cv2.VideoCapture('testvideo_3.mp4') 
  
while True: 
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 75, 150)
   lines = cv2.HoughLines(edges,2,np.pi/180,100, min_theta=-0.2,max_theta=0.2)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
     
    cv2.imshow("Edges", edges)
    cv2.imshow("Image", frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break