import cv2
import utils
  
# Replace the below URL with your own. Make sure to add "/video" at last.
url = "http://192.168.1.162:8080/video"
  
# While loop to continuously fetching data from the Url
cap = cv2.VideoCapture(url)

while(True):
    ret, frame = cap.read()
    out = cv2.transpose(frame)
    out = cv2.flip(out,flipCode=1)
    imgResized = utils.resize(out,width=600,height=utils.imgHeightPrefix)
    cv2.imshow('frame',imgResized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    imgC, contours = utils.getContours(imgResized)
    if len(contours) != 0:
        print("Contours found: %i" % len(contours))
        biggestContour = contours[0][2]
        imgWarp = utils.warpImg(imgC,biggestContour,utils.widthP,utils.heightP)
        imgContours2, contours2 = utils.getContours(imgWarp,
            minArea=2000,
            cannyT=[50,50])
        if len(imgContours2) != 0:
            for cnt in contours2:
                cv2.polylines(imgContours2,[cnt[2]],True,(0,255,0),2)
                nPoints = utils.assignPoints(cnt[2],2)
                nW = round((utils.findDis(nPoints[0][0]/utils.scaleF,nPoints[1][0]/utils.scaleF)/11),1)
                nH = round((utils.findDis(nPoints[0][0]/utils.scaleF,nPoints[2][0]/utils.scaleF)/11),1)
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
        else:
            break 
        
    #print("Contours found: %i" % len(contours2))
cv2.destroyAllWindows()
