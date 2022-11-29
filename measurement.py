import cv2
import utils

webcam = False
path = "IMG_20221128_174604.jpg"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BRIGHTNESS,160)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
scaleResize = 40
scaleF = 3
widthP = 210 * scaleF
heightP = 297 * scaleF



if webcam:success, img = cap.read()
else: img = cv2.imread(path)

imgResized = utils.resize(img, scaleResize)
width, height = imgResized.shape[:2]

imgC, contours = utils.getContours(imgResized,minArea=10000,filter=4)

if len(contours) != 0:
    biggest = contours[0][2]
    imgWarp = utils.warpImg(imgC,biggest,widthP,heightP)
    cv2.imshow("Paper",imgWarp)
else:print(len(contours))

cv2.waitKey(1)
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

