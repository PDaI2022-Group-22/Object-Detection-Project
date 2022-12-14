import cv2
import ObjectMeasurementUtils
  
# Replace the below URL with your own. Make sure to add "/video" at last.
  
# While loop to continuously fetching data from the Url

def ObjectMeasurement(url,widthMM,heightMM,):
    scaleMultiplier = 2
    widthScaled = widthMM * 2
    heightScaled = heightMM * 2
    cap = cv2.VideoCapture(url)
    while(True):
        ret, frame = cap.read()
        out = cv2.transpose(frame)
        out = cv2.flip(out,flipCode=1)
        imgResized = ObjectMeasurementUtils.resize(out,width=600,height=ObjectMeasurementUtils.imgHeightPrefix)
        cv2.imshow('frame',imgResized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        imgC, contours = ObjectMeasurementUtils.getContours(imgResized)
        if len(contours) != 0:
            print("Contours found: %i" % len(contours))
            biggestContour = contours[0][2]
            imgWarp = ObjectMeasurementUtils.warpImg(imgC,biggestContour,widthScaled,heightScaled)
            imgContours2, contours2 = ObjectMeasurementUtils.getContours(imgWarp,
                minArea=2000,
                cannyT=[50,50])
            if len(imgContours2) != 0:
                for cnt in contours2:
                    cv2.polylines(imgContours2,[cnt[2]],True,(0,255,0),2)
                    nPoints = ObjectMeasurementUtils.assignPoints(cnt[2],2)
                    nW = round((ObjectMeasurementUtils.findDis(nPoints[0][0]/scaleMultiplier,nPoints[1][0]/scaleMultiplier)/10),1)
                    nH = round((ObjectMeasurementUtils.findDis(nPoints[0][0]/scaleMultiplier,nPoints[2][0]/scaleMultiplier)/10),1)
                    cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                    (255, 0, 255), 3, 8, 0, 0.05)
                    x, y, w, h = cnt[3]
                    cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                (255, 0, 255), 2)
                    cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                (255, 0, 255), 2)
                    cv2.imshow('A4', imgContours2)
            else:
                ObjectMeasurement(url,widthMM,heightMM,)
            
        #print("Contours found: %i" % len(contours2))
    cv2.destroyAllWindows()
