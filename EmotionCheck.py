import cv2
from deepface import DeepFace

faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Jokin estää web-kameran toiminnan")

while True:
        ret,frame = cap.read()
        result = DeepFace.analyze(frame, actions= ['emotion'])

        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces= faceCascade.detectMultiScale(gray,1.1,4)

        for(x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame,
           result['dominant_emotion'],
           (50,50),
           font, 3,
           (0,255,0),
           2,
           cv2.LINE_4)
 
        cv2.imshow('Original video',frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
        