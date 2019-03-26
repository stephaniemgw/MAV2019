import pandas as pd
import matplotlib.pyplot as plt
from math import cos
from math import sin
import numpy as np

# min(xposnl) = -2.7302
# max(xposnl) = 2.7755
# min(yposnl) = -3.0116
# max(yposnl) = 2.27769



theta = 1*(np.pi/180)*32
results = []
xposl = []
yposl = []
xposnl = []
yposnl = []
xposrotatedl = []
yposrotatedl = []
safepoint_l = []

with open('safeflightlog.txt') as inputfile:
    for line in inputfile:
        results.append(line.strip().split(','))
    
for i in range(len(results)):
    line = str(results[i])
    sec = line[2:67]
    #print(sec)
    if 1 == 1:
        if sec == "[orange_avoider->orange_avoider_periodic()] Current SAFE_POS_NORM":
            #print(line[78:100])       
            xpos_norm = float(line[78:84])
            
            ypos_norm = float(line[92:98])
            #ypos = line[87:93]
            #xpos = float(xpos)
            #ypos = float(ypos)
            xposl.append(xpos_norm)
            yposl.append(ypos_norm)
    
    sec2 = line[2:70]
    if sec2 == "[orange_avoider->orange_avoider_periodic()] Current SAFE_POS_ROTATED":
        xpos_rotated = float(line[81:87])
        ypos_rotated = float(line[95:101])
        xposrotatedl.append(xpos_rotated)
        yposrotatedl.append(ypos_rotated)
        
    # print(sec2) # [orange_avoider->orange_avoider_periodic()] SAFE GRID: D8']
    
    sec3 = line[2:55]
    if sec3 == "[orange_avoider->orange_avoider_periodic()] SAFE GRID":
        safepoint = line[57:59]
        safepoint_l.append(safepoint)
        
# SORT SAFEPOINTS
safepoint_l_sorted = set(safepoint_l)

                
        
        
#[orange_avoider->orange_avoider_periodic()] Current SAFE_POS (X,Y):
        
ROT = np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])


for j in range(len(xposl)):
    xposn = cos(theta)*xposl[j]-sin(theta)*yposl[j]
    yposn = sin(theta)*xposl[j]+cos(theta)*yposl[j]
    xposnl.append(xposn)
    yposnl.append(yposn)

#for i in range(3087):
#    print(results[i])

if 0 == 1:
    plt.plot(xposl,yposl,'ro')
    plt.axis(xmin=-5,xmax=5,ymin=-5,ymax=5)
    plt.grid()
    plt.show()
    
    plt.plot(xposnl,yposnl,'bo')
    plt.axis(xmin=-5,xmax=5,ymin=-5,ymax=5)
    plt.grid()

print("Minimum x flown is:",min(xposrotatedl))
print("Maximum x flown is:",max(xposrotatedl))
print("Minimum y flown is:",min(yposrotatedl))
print("Maximum y flown is:",max(yposrotatedl))

Ngrid = 8
Xmingrid = -2.75
Xmaxgrid = 2.75
Ymingrid = -3
Ymaxgrid = 2.5

grid_int_x = (Xmaxgrid-Xmingrid)/Ngrid
grid_int_y = (Ymaxgrid-Ymingrid)/Ngrid

namesl = []
x_centsl = []
y_centsl = []
x_centsl_safe = []
y_centsl_safe = []
x_leftsl = []
y_downsl = []

alf = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

for x_i in range(Ngrid):
    x_cent = round((Xmingrid+0.5*grid_int_x)+x_i*grid_int_x,3)
    x_left = round((Xmingrid)+x_i*grid_int_x,3)
    for y_i in range(Ngrid):
        y_cent = round((Ymingrid+0.5*grid_int_y)+y_i*grid_int_y,3)
        y_down = round((Ymingrid)+y_i*grid_int_y,3)
        
        name = str(alf[x_i])+str(y_i+1)
        
        # CHECK IF GRID IS SAFE
        if name in safepoint_l_sorted:
            x_centsl_safe.append(x_cent)
            y_centsl_safe.append(y_cent)
        
        
        namesl.append(name)
        x_centsl.append(x_cent)
        y_centsl.append(y_cent)
        x_leftsl.append(x_left)
        y_downsl.append(y_down)




slash = "\n"
if 2 == 2:
    for itera in range(len(namesl)):
        vartekst = "if(x_pos_safe_rotated>"+str(x_leftsl[itera])+" && x_pos_safe_rotated<"+str(round((x_leftsl[itera]+grid_int_x),3))+ " && y_pos_safe_rotated>"+str(y_downsl[itera])+" && y_pos_safe_rotated<"+str(round((y_downsl[itera]+grid_int_y),3))+"){"
        print(vartekst)
        #print("{") # REMOVED FOR ALTERNATIVE NOTATION
        print('    VERBOSE_PRINT("SAFE GRID: '+namesl[itera]+'\\n");')
        print("}")
        
        
        
### PLOT PART
        
#plt.plot(xposl,yposl,'ro',label="unrotated") # NORM
plt.plot(xposrotatedl,yposrotatedl,'bo',label="rotated in C") # ROTATED IN C
plt.plot(x_centsl,y_centsl,'ro',label="mid grid-points")
plt.plot(x_centsl_safe,y_centsl_safe,'go',label="safe grid points")
plt.legend()