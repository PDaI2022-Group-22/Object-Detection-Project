import cv2
import numpy as np
import math
from scipy import ndimage
from matplotlib import pyplot as plt


path = './images/angles.jpg'
image = cv2.imread(path)
centerPoint = (int(image.shape[0]/2),int(image.shape[1]/2))
cv2.circle(image,(centerPoint),5,(0,0,255),cv2.FILLED)
pointsList = []

def mousePoints(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
            cv2.line(image,tuple(pointsList[round((size-1)/3)*3]),(x,y),(0,0,255),2)
        cv2.circle(image,(x,y),5,(0,0,255),cv2.FILLED)
        pointsList.append([x,y])
        
        

def gradient(pt1,pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0]+1e-5)

def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = math.atan((m2-m1)/(1 +(m2*m1)))
    #90 - 180
    if angR < 0 and pt2[1] < pt1[1]:
        angRCorrected = math.pi + angR
        angD = abs(round(math.degrees(angRCorrected)))
    #180 - 270
    elif angR > 0 and pt2[1] > pt1[1]:
        angRCorrected = math.pi + angR
        angD = abs(round(math.degrees(angRCorrected)))
    #270 - 360
    elif angR < 0 and pt2[0] > pt1[0]:
        angRCorrected = (2*math.pi) + angR
        angD = abs(round(math.degrees(angRCorrected)))
    #0 - 90
    else:
        angD = abs(round(math.degrees(angR)))

    print("Gradient 1 : %f\nGradient 2: %f\nAngle with abs() : %i" %  (m1, m2, angD))
    print("Angle rads : %f" % (angR))
    print("PTX [ X , Y ]")
    print("PT1 [%i,%i]\nPT2 [%i,%i]\nPT3 [%i,%i]\n"  
    % (pt1[0], pt1[1], pt2[0],pt2[1], pt3[0],pt3[1]))

    cv2.putText(image, "PT1",(pt1[0]+20, pt1[1]+20),cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(0,0,0), 1)
    cv2.putText(image, "PT2",(pt2[0]-20, pt2[1]-20),cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(0,0,0), 1)
    cv2.putText(image, "PT3",(pt3[0]-20, pt3[1]-20),cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(0,0,0), 1)
    cv2.putText(image,str(angD),(pt1[0]-40, pt1[1]-20),cv2.FONT_HERSHEY_COMPLEX,
                1.5,(0,0,0), 1)

while True:

    if len(pointsList) != 0 and len(pointsList) % 3 == 0:
        getAngle(pointsList)

    cv2.imshow("window",image)
    cv2.setMouseCallback('window', mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointsList = []
        image = cv2.imread(path)