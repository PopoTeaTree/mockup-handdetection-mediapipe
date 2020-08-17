import cv2
import numpy as np
import random
import pyautogui
from src.hand_tracker import HandTracker
from scipy.interpolate import interp1d
from numpy import interp
from scipy.spatial import distance
import glob
from PIL import Image
import os
import pandas as pd
from matplotlib import pyplot as plt


WINDOW = "Hand Tracking"
# WINDOW2 = "Rectangle Game"
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
    x = random.randint(0, xMax)
    y = random.randint(0, yMax)
    return x, y

def randomRectangle(framee,x,y):
    x = x
    y = y
    w = x+500
    h = y+300
    cv2.rectangle( framee,(x, y),(w,h),(255,0,0), 1, 8)
    return x,y,w,h

def isHandInRectngle(xP,yP,x,y,w,h):
    if x<xP<w and y<yP<h:
        return True
    else:
        return False

def mappingPointX(frameCamera,displayScreen,pointX):
    wO = frameCamera.shape[1]
    wN = displayScreen.shape[1]
    # return int((pointX * wN) / wO)
    # mX = interp1d([0,wO],[0,wN])
    # return int(mX(pointX))
    return int(interp(pointX,[0,wO],[0,wN]))

def mappingPointY(frameCamera,displayScreen,pointY):
    hO = frameCamera.shape[0]
    hN = displayScreen.shape[0]
    # return int((pointY * hN) / hO)
    # mY = interp1d([0,hO],[0,hN])
    # return int(mY(pointY))
    return int(interp(pointY,[0,hO],[0,hN]))

cv2.namedWindow(WINDOW)
# cv2.namedWindow(WINDOW2)
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

print("Frame shape: "+ str(frame.shape))
firstRound = True     # check the first round

comShape = pyautogui.size()
heightCom = comShape[1]
widthCom = comShape[0]

height = frame.shape[0]
width = frame.shape[1]
channels = frame.shape[2]

xMax = width
yMax = height

# x and y rectangle axis at the moment
xT = 0
yT = 0

while hasFrame:
    frame = cv2.flip(frame, 1)

    # y=50
    # x=50
    # h=500
    # w=300
    # crop_frame = frame[x:w, y:h]

    #Resize VDO
    # scale_percent = 60 # percent of original size
    # width = int(frame.shape[1] * scale_percent / 100)
    # height = int(frame.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # crop_frame = cv2.resize(frame,dim)
    # cv2.imshow("cropedframe",crop_frame)

    # display = np.zeros((heightCom,widthCom,channels),np.uint8)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)

    # if firstRound:
    #     xR, yR = xyrandom(xMax,yMax)
    #     randomRectangle(display,xR,yR)
    #     xT = xR
    #     yT = yR
    #     firstRound = False
    # else:
    #     randomRectangle(display,xT,yT)

    if points is not None:
        # newPoints = []
        # # mapping new xy point
        # for point in points:
        #     x, y = point
        #     newPoint = []
        #     # newX = mappingPointX(frame,display,x)
        #     # newY = mappingPointY(frame,display,y)
        #     newX = mappingPointX(crop_frame,display,x)
        #     newY = mappingPointY(crop_frame,display,y)
        #     newPoint.append(newX)
        #     newPoint.append(newY)
        #     newPoints.append(newPoint)
        # # print("Old point: " + str(points))
        # # print("New point: " + str(newPoints))

        # track the first finger points[8]
        x0, y0 = points[0]
        cv2.circle(frame, (int(x0), int(y0)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        cv2.putText(frame, str(int(x0))+","+str(int(y0)), (int(x0)+4, int(y0)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.circle(frame, (10, 55), THICKNESS * 2, POINT_COLOR, THICKNESS)

        # xP = mappingPointX(crop_frame,display,x0)
        # yP = mappingPointY(crop_frame,display,y0)
        # # xP, yP = newPoints[0]
        # # cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        # # cv2.putText(frame, str(int(x))+","+str(int(y)), (int(x)+4, int(y)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        # cv2.circle(display, (int(xP), int(yP)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        # cv2.putText(display, str(int(xP))+","+str(int(yP)), (int(xP)+4, int(yP)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER, distanceFinger = fingerState_distance_ratio(points)
        # isClick = gestureClick(MIDDLEFINGER, RINGFINGER, LITTLEFINGER,display)

        # cv2.rectangle(display,(200,200),(300,300),(255,0,0), 1, 8)

        # isPointIn = isHandInRectngle(xP,yP,xT,yT,xT+500,yT+300)

        # if isPointIn and isClick:
        #     cv2.putText(display, "Catch the rectangle", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
        #     xR, yR = xyrandom(xMax,yMax)
        #     xT = xR
        #     yT = yR

    # cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    cv2.imshow(WINDOW,frame)
    # cv2.imshow(WINDOW2,display)
    # cv2.imshow("cropedframe",crop_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()