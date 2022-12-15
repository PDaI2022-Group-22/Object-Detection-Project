import os
import random
import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import cvzone

def TasteRain():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = FaceMeshDetector(maxFaces=1)
    idList = [0, 17, 78, 292]

    # import images
    folderEatable = 'Objects\\eatable'
    listEatable = os.listdir(folderEatable)
    eatables = []
    for object in listEatable:
        eatables.append(cv2.imread(f'{folderEatable}/{object}', cv2.IMREAD_UNCHANGED))

    folderLightning = 'Objects\\lightning'
    listLightning = os.listdir(folderLightning)
    Lightning = []
    for object in listLightning:
        Lightning.append(cv2.imread(f'{folderLightning}/{object}', cv2.IMREAD_UNCHANGED))

    folderNonEatable = 'Objects\\noneatable'
    listNonEatable = os.listdir(folderNonEatable)
    nonEatables = []
    for object in listNonEatable:
        nonEatables.append(cv2.imread(f'{folderNonEatable}/{object}', cv2.IMREAD_UNCHANGED))

    folderCritters = 'Objects\\critters'
    listCritters = os.listdir(folderCritters)
    Critters = []
    for object in listCritters:
        Critters.append(cv2.imread(f'{folderCritters}/{object}', cv2.IMREAD_UNCHANGED))

    currentObject = eatables[0]
    pos = [300, 0]
    speed = 5
    count = 0
    global isCritter
    global isEatable
    isCritter= False
    isEatable = True
    isLightning = False
    gameOver = False


    def resetObject():
        global isEatable
        global isCritter
        global isLightning
        pos[0] = random.randint(100, 1180)
        pos[1] = 0
        randNo = random.randint(0, 4)  # ratio of eatables/ non-eatables
        if randNo == 0:
            currentObject = nonEatables[random.randint(0, 3)]
            isEatable = False
            isCritter = False
            isLightning = False
        elif  randNo == 1  and count>=10:
            currentObject = Critters[random.randint(0, 3)]
            isEatable = False
            isCritter = True
            isLightning = False
        elif randNo == 2  and count>=5:
            currentObject = Lightning[random.randint(0, 0)]
            isEatable = False
            isLightning = True
        else:
            currentObject = eatables[random.randint(0, 3)]
            isEatable = True
            isCritter = False
            isLightning = False

        return currentObject



    while True:
        success, img = cap.read()

        if gameOver is False:
            img, faces = detector.findFaceMesh(img, draw=False)

            img = cvzone.overlayPNG(img, currentObject, pos)
            if isCritter==True:
                if cx>=pos[0]:
                    pos[1] += (speed + count)
                    pos[0] += (5 + count)
                else:
                    pos[1] += (speed + count)
                    pos[0] -= (5 + count)
            if isLightning==True:
                pos[1] += (count + count)
            else:
                pos[1] += (speed + count)   

        # img = cvzone.overlayPNG(img, currentCritter, pos)
            #pos[2] += speed/5
            if pos[1] > 600:
                currentObject = resetObject()
            
            if pos[0] > 1180 or pos[0] < 99:
                currentObject = resetObject()

            if faces:
                face = faces[0]
    
                up = face[idList[0]]
                down = face[idList[1]]

                for id in idList:
                    cv2.circle(img, face[id], 5, (255, 255, 255), 5)
                cv2.line(img, up, down, (0, 255, 0), 3)
                cv2.line(img, face[idList[2]], face[idList[3]], (0, 255, 0), 3)

                upDown, _ = detector.findDistance(face[idList[0]], face[idList[1]])
                leftRight, _ = detector.findDistance(face[idList[2]], face[idList[3]])

                cx, cy = (up[0] + down[0]) // 2, (up[1] + down[1]) // 2
                cv2.line(img, (cx, cy), (pos[0] + 50, pos[1] + 50), (0, 255, 0), 3)
                distMouthObject, _ = detector.findDistance((cx, cy), (pos[0] + 50, pos[1] + 50))
                print(distMouthObject)
                print(cx)
                

                ratio = int((upDown / leftRight) * 100)
                if ratio > 60:
                    mouthStatus = "Open"
                else:
                    mouthStatus = "Closed"
                cv2.putText(img, mouthStatus, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

                if distMouthObject < 100 and ratio > 60:
                    if isEatable:
                        currentObject = resetObject()
                        count += 1
                    else:
                        gameOver = True
            cv2.putText(img, str(count), (1100, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)
        else:
            cv2.putText(img, "Game Over", (300, 400), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 255), 10)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)

        if key == ord('r'):
            resetObject()
            gameOver = False
            count = 0
            currentObject = eatables[0]
            isEatable = True

        if key == ord('q'):   
            break
    cap.release()
    cv2.destroyAllWindows()

