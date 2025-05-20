import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Variables
width, height = 1280, 720
folderPath = "presentation"

# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get the list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# Variables
imgNumber = 1
hs, ws = 120, 213  # webcam corner size
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 30
annotations = [[]]
annotationNumber = 0
annotationStart = False

# Hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    # Capture webcam images
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Load current presentation image
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    imgCurrent = cv2.resize(imgCurrent, (width, height))  # Ensure same size

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)
    print(annotationNumber)

    if hands and  buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        # Get index finger position
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))
        indexFinger = (xVal, yVal)

        if cy <= gestureThreshold:
            # Gesture 1: Previous Slide
            annotationStart=False
            if fingers == [1, 0, 0, 0, 0]:
                annotationStart=False
                print("Left")
                if imgNumber > 0:
                    buttonPressed = True
                    annotations=[[]]
                    annotationNumber=0
                    imgNumber -= 1

            # Gesture 2: Next Slide
            if fingers == [0, 0, 0, 0, 1]:
                annotationStart=False
                print("Right")
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber += 1

        # Gesture 3: Show Pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotationStart = False

        # Gesture 4: Draw
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)

        else:
            annotationStart = False

        # Gesture 5: Erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                if annotationNumber >= 0:
                    annotations.pop()
                    annotationNumber -= 1
                    buttonPressed = True
    else:
        annotationStart=False


    # Handle delay for button presses
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    # Draw Annotations
    for i in range(len(annotations)):
        for j in range(1, len(annotations[i])):
            cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

    # Add webcam feed to slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    # Show images
    cv2.imshow("Webcam", img)
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()