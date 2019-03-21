# capture frames from a camera 
import cv2
import numpy as np

cap = cv2.VideoCapture('testvideo_3.mp4') 
  
while True: 
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    edges = cv2.Canny(gray, 100, 200)
    # use n=200 voor palen
    # use n=130 voor ook lijsten, is wel minder accuraat
    lines = cv2.HoughLines(edges,1,np.pi/180,200, min_theta=-0.2,max_theta=0.2)
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
        
                cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
 
    cv2.imshow("Edges", edges)
    cv2.imshow("Image", frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
