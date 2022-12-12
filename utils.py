import cv2
import numpy

#constants
measurementBGColor = "Black"
imgHeightPrefix = 720
scaleF = 2
widthP = 275 * scaleF
heightP = 320 * scaleF


def resize(image, width = None, height = None, inter = cv2.INTER_AREA, save=False):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # ability to save the resized image.
    if save == True:
        cv2.imwrite("resized.png",resized)
    # return the resized image
    return resized

def getContours(img,cannyT=[50,180],display=False,minArea=50000,draw=False):
    kernel = numpy.ones((5,5))
    contoursList = []
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(11,11),0)
    imgCanny = cv2.Canny(imgBlur, cannyT[0], cannyT[1], 3)
    imgDilated = cv2.dilate(imgCanny, kernel, iterations=3)
    imgEroded = cv2.erode(imgDilated, kernel,iterations=2)

    if display == True:
        cv2.imshow("Eroded",imgEroded)

    contours, _ = cv2.findContours(imgDilated, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        
        if area > minArea:
            perimtr = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c, 0.02*perimtr, True)
            bbox = cv2.boundingRect(approx)
            contoursList.append([len(approx),area,approx,bbox,c])
            contoursList = sorted(contoursList, key = lambda x:x[1],reverse=True)
        if draw:
            for cont in contoursList:
                cv2.drawContours(img,cont[4], -1,(0,0,255),3)
    
    return img, contoursList

def assignPoints(pointsList,method=1):
    # 2 different methods for assigning points either clock-wise or counter clock-wise.
    if method == 1:
        pointsOrig = pointsList.reshape((4,2))
        pointsToList = pointsOrig.tolist()
        pointsIndexed = []
        pt_A = pointsToList[0]
        pt_B = pointsToList[1]
        pt_C = pointsToList[2]
        pt_D = pointsToList[3]
        pointsIndexed.extend([pt_A,pt_B,pt_C,pt_D])
        return pointsIndexed
    elif method == 2 or method >= 2:
        myPointsNew = numpy.zeros_like(pointsList)
        pointsAmnt = len(pointsList)
        pointsList = pointsList.reshape((pointsAmnt,2))
        add = pointsList.sum(1)
        myPointsNew[0] = pointsList[numpy.argmin(add)]
        myPointsNew[3] = pointsList[numpy.argmax(add)]
        diff = numpy.diff(pointsList,axis=1)
        myPointsNew[1]= pointsList[numpy.argmin(diff)]
        myPointsNew[2] = pointsList[numpy.argmax(diff)]
        return myPointsNew

def warpImg(img,biggestContour,w,h):
    points = assignPoints(biggestContour,2)
    pts1 = numpy.float32(points)
    pts2 = numpy.float32([[0,0],[w,0],[0,h],[w,h]])
    mtx = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,mtx,(w,h),flags=cv2.INTER_LINEAR)
    return imgWarp

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5
    
