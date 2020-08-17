import cv2
import numpy as np
import random
import os
import pyautogui
from src.hand_tracker import HandTracker
from scipy.interpolate import interp1d
from numpy import interp
from scipy.spatial import distance
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import statistics

WINDOW = "Hand Tracking"
WINDOW2 = "Rectangle Game"
PALM_MODEL_PATH = "models/palm_detection_builtin.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2



def gestureClick(MIDDLEFINGER, RINGFINGER, LITTLEFINGER,frame):
    isClick = False
    if not MIDDLEFINGER and not RINGFINGER and not LITTLEFINGER:
        cv2.putText(frame, "Close", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        return True
    if MIDDLEFINGER and RINGFINGER and LITTLEFINGER:
        cv2.putText(frame, "Open", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    return isClick

def finger_state(points):
    # finger state
    THUMB = False
    INDEXFINGER = False
    MIDDLEFINGER = False
    RINGFINGER = False
    LITTLEFINGER = False

    pseudoFixKeyPoint1 = points[2][0]
    if points[3][0] < pseudoFixKeyPoint1 and points[4][0] < pseudoFixKeyPoint1:
        THUMB = True
        # print("Trumb UP")
        # cv2.putText(frame, "Trumb UP", (points[2][0], points[2][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    pseudoFixKeyPoint2 = points[6][1]
    if points[7][1] < pseudoFixKeyPoint2 and points[8][1] < pseudoFixKeyPoint2:
        INDEXFINGER = True
        # print("INDEXFINGER")
        # cv2.putText(frame, "INDEXFINGER", (points[6][0], points[6][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    pseudoFixKeyPoint3 = points[10][1]
    if points[11][1] < pseudoFixKeyPoint3 and points[12][1] < pseudoFixKeyPoint3:
        MIDDLEFINGER = True
        # print("MIDDLEFINGER")
        # cv2.putText(frame, "MIDDLEFINGER", (points[10][0], points[10][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    pseudoFixKeyPoint4 = points[14][1]

    if points[15][1] < pseudoFixKeyPoint4 and points[16][1] < pseudoFixKeyPoint4:
        RINGFINGER = True
        # print("RINGFINGER")
        # cv2.putText(frame, "RINGFINGER", (points[14][0], points[14][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    pseudoFixKeyPoint5 = points[18][1]
    if points[19][1] < pseudoFixKeyPoint5 and points[20][1] < pseudoFixKeyPoint5:
        LITTLEFINGER = True
        # print("LITTLEFINGER")
        # cv2.putText(frame, "LITTLEFINGER", (points[20][0], points[20][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    return INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER

def fingerState_distance_Compare(points):
    # finger state
    # THUMB = False
    INDEXFINGER = False
    MIDDLEFINGER = False
    RINGFINGER = False
    LITTLEFINGER = False
    # Tumb
    pseudoFixKeyPoint1 = points[2][0]
    if points[3][0] < pseudoFixKeyPoint1 and points[4][0] < pseudoFixKeyPoint1:
        THUMB = True
    # Index finger
    if distance.euclidean(points[0], points[8]) > distance.euclidean(points[0], points[5]):
        INDEXFINGER = True
    # Middle finger
    if distance.euclidean(points[0], points[12]) > distance.euclidean(points[0], points[9]):
        MIDDLEFINGER = True
    # Ring finger
    if distance.euclidean(points[0], points[16]) > distance.euclidean(points[0], points[13]):
        RINGFINGER = True
    # Little finger
    if distance.euclidean(points[0], points[20]) > distance.euclidean(points[0], points[17]):
        LITTLEFINGER = True
    return INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER

def fingerState_distance_ratio(points):
    distanceFinger = []
    # finger state
    # THUMB = False
    INDEXFINGER = False
    MIDDLEFINGER = False
    RINGFINGER = False
    LITTLEFINGER = False
    # Tumb
    pseudoFixKeyPoint1 = points[2][0]
    if points[3][0] < pseudoFixKeyPoint1 and points[4][0] < pseudoFixKeyPoint1:
        THUMB = True
    # Index finger
    # print( "Index: " + str(distance.euclidean(points[0], points[8])/distance.euclidean(points[0], points[5])) )
    distanceFinger.append(distance.euclidean(points[0], points[8])/distance.euclidean(points[0], points[5]))
    if distance.euclidean(points[0], points[8])/distance.euclidean(points[0], points[5]) > 1.5 :
        INDEXFINGER = True
    # Middle finger
    # print( "Middle: " + str(distance.euclidean(points[0], points[12])/distance.euclidean(points[0], points[9])))
    distanceFinger.append(distance.euclidean(points[0], points[12])/distance.euclidean(points[0], points[9]))
    if distance.euclidean(points[0], points[12])/distance.euclidean(points[0], points[9]) > 1.8 :
        MIDDLEFINGER = True
    # Ring finger
    # print( "Ring: " + str(distance.euclidean(points[0], points[16])/distance.euclidean(points[0], points[13])))
    distanceFinger.append(distance.euclidean(points[0], points[16])/distance.euclidean(points[0], points[13]))
    if distance.euclidean(points[0], points[16])/distance.euclidean(points[0], points[13]) > 1.6 :
        RINGFINGER = True
    # Little finger
    # print( "little: " + str(distance.euclidean(points[0], points[20])/distance.euclidean(points[0], points[17])))
    distanceFinger.append(distance.euclidean(points[0], points[20])/distance.euclidean(points[0], points[17]))
    if distance.euclidean(points[0], points[20])/distance.euclidean(points[0], points[17]) > 1.56 :
        LITTLEFINGER = True
    return INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER, distanceFinger
    # return distanceFinger

def xyrandom(xMax,yMax):
    x = random.randint(100, xMax)
    y = random.randint(100, yMax)
    return x, y

def randomRectangle(framee,x,y):
    w = x
    h = y
    cv2.rectangle( framee,(x, y),(w,h),(255,0,0), 1, 8)
    return x,y,w,h

def isHandInRectngle(xP,yP,x,y,w,h):
    if x<xP<w and y<yP<h:
        return True
    else:
        return False

def mappingPointX(wO,wN,pointX):
    # return int(mX(pointX))
    # return int(interp(pointX,[0,wO],[0,wN]))
    # return int((pointX * wN) / wO)
    return int(( (pointX) / (wO) ) * (wN))

def mappingPointY(hO,hN,pointY):
    # return int(mY(pointY))
    # return int(interp(pointY,[0,hO],[0,hN]))
    # return int((pointY * hN) / hO)
    return int(( (pointY) / (hO) ) * (hN))

def mappingPoint(OldMax,OldMin,NewMax,NewMin,OldValue):
    return int( (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin )

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

comShape = pyautogui.size()
print("Frame shape: "+ str(frame.shape))
print("Com shape: "+ str(comShape))

channels = frame.shape[2]

heightCom = comShape[1]
widthCom = comShape[0]

counter = 0
pointDataX = []
pointDataY = []

while hasFrame:
    frame = cv2.flip(frame, 1)

    # crop frame to map screen
    yCrop=230
    xCrop=155
    hCrop=390
    wCrop=480
    crop_frame = frame[yCrop:hCrop,xCrop:wCrop]
    cv2.rectangle(frame,(xCrop,yCrop),(wCrop,hCrop),(255,255,0), 1, 8)

    # plt.imshow(np.asarray(frame))
    # plt.show()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)

    if points is not None:
        newPoints = []
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            # cv2.putText(frame, str(int(x))+","+str(int(y)), (int(x)+4, int(y)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        # print(str(distance.euclidean(points[5], points[8]))+" , "+str(distance.euclidean(points[0], points[8])))

        # clicking point shows on camera screen
        x0, y0 = points[0]
        cv2.circle(frame, (int(x0), int(y0)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        cv2.putText(frame, str(int(x0))+","+str(int(y0)), (int(x0)+4, int(y0)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # connection camera screen
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

        # track the first finger points[8]
        xP, yP = points[0]
        # Check is the wrist point on the rectanger
        isPointIn = isHandInRectngle(xP,yP,xCrop,yCrop,wCrop,hCrop)
        print("Crop frame shape: " + str(crop_frame.shape[1]))
        if isPointIn:
            print("==========In==============")
            # map frame wrist to crop
            xCroped= mappingPoint(wCrop,xCrop,crop_frame.shape[1],0,xP)
            yCroped = mappingPoint(hCrop,yCrop,crop_frame.shape[0],0,yP)
            print("mapping point crop frame: " + str(xCroped)+","+str(yCroped))
            # map wrist to screen
            xScreen= mappingPoint(crop_frame.shape[1],0,widthCom,0,xCroped)
            yScreen = mappingPoint(crop_frame.shape[0],0,heightCom,0,yCroped)
            print("mapping point screen: " + str(xScreen)+","+str(yScreen))

            pointDataX.append(xScreen)
            pointDataY.append(yScreen)

            if(len(pointDataX) == 3 and len(pointDataY)==3):
                xMouse = int(statistics.mean(pointDataX))
                yMouse = int(statistics.mean(pointDataY))
                # moving mouse
                pyautogui.moveTo(xMouse, yMouse, duration = 0)
                # Checking gesture clicking
                INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER, distanceFinger = fingerState_distance_ratio(points)
                isClick = gestureClick(MIDDLEFINGER, RINGFINGER, LITTLEFINGER,frame)
                if isClick:
                    pyautogui.click(xMouse, yMouse)
                pointDataX = []
                pointDataY = []
            else:
                pass
        else:
            pass

    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()