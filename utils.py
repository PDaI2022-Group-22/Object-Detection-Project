import cv2
import sys
import numpy

def getResolution(imgShape):
    pass

def resize(img,scaleP):
    height =int(img.shape[0] * scaleP / 100) 
    width = int(img.shape[1] * scaleP / 100)
    dsize = (height,width)
    imgResized = cv2.resize(img,dsize)
    return imgResized

def getContours(img,cannyT=[100,100],display=False,minArea=1000,filter=0,draw=False):
    # creating a 5*5 array of 1's
    kernel = numpy.ones((5,5))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur, cannyT[0], cannyT[1])
    imgDilated = cv2.dilate(imgCanny, kernel, iterations=3)
    imgEroded = cv2.erode(imgDilated, kernel, iterations=2)
    if display == True:cv2.imshow("Threshold",imgEroded)

    cnts = cv2.findContours(imgEroded, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

    contoursList = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > minArea:
            perimtr = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c, 0.02*perimtr, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    contoursList.append([len(approx),area,approx,bbox,c])
            else:
                contoursList.append([len(approx),area,approx,bbox,c])
    contoursList = sorted(contoursList, key = lambda x:x[1],reverse=True)
    if draw:
        for cont in contoursList:
            cv2.drawContours(img,cont[4], -1,(0,0,255),3)
    
    return img, contoursList



def assignPoints(pointsList):
    pointsOrig = pointsList.reshape((4,2))
    pointsToList = pointsOrig.tolist()
    pointsIndexed = []
    pt_A = pointsToList[0]
    pt_B = pointsToList[1]
    pt_C = pointsToList[2]
    pt_D = pointsToList[3]
    pointsIndexed.extend([pt_A,pt_B,pt_C,pt_D])
    return pointsIndexed


def calcWidthHeight(pointsList):
    if len(pointsList) != 0:
        pt_A, pt_B, pt_C, pt_D = assignPoints(pointsList)
        print(pt_A[0])
        print(pt_B[1])
        
        width_AD = numpy.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
        width_BC = numpy.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
        maxWidth = max(int(width_AD), int(width_BC))

        height_AB = numpy.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
        height_CD = numpy.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
        maxHeight = max(int(height_AB), int(height_CD))
    return maxWidth, maxHeight


def midpoint(p1, p2):
    p1X, p1Y = p1
    p2X, p2Y = p2
    middleX = (p1X + p2X) / 2
    middleY = (p1Y + p2Y) / 2
    midpoint = middleX, middleY
    return midpoint

def warpImg(img,biggestC,w,h):
    maxWidth, maxHeight = calcWidthHeight(biggestC)
    pts1 = numpy.float32(biggestC)
    pts2 = numpy.float32([[0,0],[0,maxHeight-1],[maxWidth-1,maxHeight-1],[maxWidth-1,0]])
    mtx = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,mtx,(maxWidth,maxHeight),flags=cv2.INTER_LINEAR)
    return imgWarp
