import cv2
import utils

path = "IMG_20221201_141524.jpg"
imgHeightPrefix = 720
scaleF = 2
widthP = 275 * scaleF
heightP = 320 * scaleF

img = cv2.imread(path)

imgResized = utils.resize(img,height=imgHeightPrefix)
width, height = imgResized.shape[:2]

imgC, contours = utils.getContours(imgResized,display=True)

if len(contours) != 0:
    print("Contours found: %i" % len(contours))
    biggestContour = contours[0][2]
    imgWarp = utils.warpImg(imgC,biggestContour,widthP,heightP)
    cv2.imshow("TEST",imgWarp)
    cv2.waitKey(0)
else:print("Contours not found!")
    
''' 
    imgContours2, contours2 = utils.getContours(imgWarp,
        maxArea=2000,
        cannyT=[50,50],draw = True)

    if len(contours2) != 0:
        for cnt in contours2:
            cv2.polylines(imgContours2,[cnt[2]],True,(0,255,0),2)
            nPoints = utils.assignPoints(cnt[2],2)
            nW = round((utils.findDis(nPoints[0][0]//scaleF,nPoints[1][0]//scaleF)/10),1)
            nH = round((utils.findDis(nPoints[0][0]//scaleF,nPoints[2][0]//scaleF)/10),1)
            cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            x, y, w, h = cnt[3]
            cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
            cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
            cv2.imshow('A4', imgContours2)
 '''



#NOT MEASURING OBJECTS UNTIL OBJECT WARPING IS WORKING
"""
# looping through contours
c = 0
while c < 3:
    # getting the bounding rectangle
    x,y,w,h = cv2.boundingRect(cnt[c])
    # find minimum area

    rect = cv2.minAreaRect(cnt[c])
    (x, y), (w, h), angle = rect

    # calculate coordinates of the minimum area rectangle
    box = cv2.boxPoints(rect)
    corners = cv2.boxPoints(rect)

    # normalize coordinates to integers
    box = np.int0(box)
    (tl, tr, br, bl) = box

    # compute the midpoint between the top-left and top-right points
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
	
	  # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

	  # draw the midpoints on the image
    cv2.circle(resized, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(resized, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(resized, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(resized, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the mid points
    cv2.line(resized, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
    cv2.line(resized, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)
    
    # calculating distances between the midpoints tuple (x,y)
    dA = math.dist((tltrX, tltrY), (blbrX, blbrY))
    dB = math.dist((tlblX, tlblY), (trbrX, trbrY))

    # Calculation of height dimA
    dimA = dA / pixelsPerCm
    # Calculation of width dimA
    dimB = dB / pixelsPerCm

    cv2.putText(resized, "{:.1f}cm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
    cv2.putText(resized, "{:.1f}cm".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
    # draw contours
    cv2.drawContours(resized, [box], 0, (0,0, 255), 1)
    c += 1
"""    

