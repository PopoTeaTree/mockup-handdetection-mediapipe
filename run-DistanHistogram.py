import cv2
import numpy as np
import random
from src.hand_tracker import HandTracker
from scipy.interpolate import interp1d
from numpy import interp
from scipy.spatial import distance
import glob
from PIL import Image
import os
import pandas as pd
from matplotlib import pyplot as plt
import ast
import seaborn as sns
import math

WINDOW = "Hand Tracking"
WINDOW2 = "Rectangle Game"
PALM_MODEL_PATH = "models/palm_detection_builtin.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2



def gesture(INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER,frame):
    if not MIDDLEFINGER and not RINGFINGER and not LITTLEFINGER:
        cv2.putText(frame, "Close", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    if MIDDLEFINGER and RINGFINGER and LITTLEFINGER:
        cv2.putText(frame, "Open", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

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
    if distance.euclidean(points[0], points[8])/distance.euclidean(points[0], points[5]) > 1 :
        INDEXFINGER = True
    # Middle finger
    # print( "Middle: " + str(distance.euclidean(points[0], points[12])/distance.euclidean(points[0], points[9])))
    distanceFinger.append(distance.euclidean(points[0], points[12])/distance.euclidean(points[0], points[9]))
    if distance.euclidean(points[0], points[12])/distance.euclidean(points[0], points[9]) > 1 :
        MIDDLEFINGER = True
    # Ring finger
    # print( "Ring: " + str(distance.euclidean(points[0], points[16])/distance.euclidean(points[0], points[13])))
    distanceFinger.append(distance.euclidean(points[0], points[16])/distance.euclidean(points[0], points[13]))
    if distance.euclidean(points[0], points[16])/distance.euclidean(points[0], points[13]) > 1 :
        RINGFINGER = True
    # Little finger
    # print( "little: " + str(distance.euclidean(points[0], points[20])/distance.euclidean(points[0], points[17])))
    distanceFinger.append(distance.euclidean(points[0], points[20])/distance.euclidean(points[0], points[17]))
    if distance.euclidean(points[0], points[20])/distance.euclidean(points[0], points[17]) > 1 :
        LITTLEFINGER = True
    # return INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER, distanceFinger
    return distanceFinger
def xyrandom():
    x = random.randint(0, 380)
    y = random.randint(0, 540)
    return x, y

def randomRectangle(framee):
    x, y = xyrandom()
    w = x+100
    h = y+100
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
    # mX = interp1d([0,wO],[0,wN])
    # return int(mX(pointX))
    return int(interp(pointX,[0,wO],[0,wN]))


def mappingPointY(frameCamera,displayScreen,pointY):
    hO = frameCamera.shape[0]
    hN = displayScreen.shape[0]
    # mY = interp1d([0,hO],[0,hN])
    # return int(mY(pointY))
    return int(interp(pointY,[0,hO],[0,hN]))

# print("Frame shape: "+ str(frame.shape))

# height = frame.shape[0]
# width = frame.shape[1]
# channels = frame.shape[2]

# read close distance text file
allDistanceFingerClose = []
fileRead = open("closeHand-Dis.txt", "r")
fileLines = fileRead.readlines()
if fileRead.mode == 'r':
    for readBuffer in fileLines:
        readBuffer = ast.literal_eval(readBuffer)
        allDistanceFingerClose.append(readBuffer)

# read open distance text file
allDistanceFingerOpen = []
fileRead = open("openHand-Dis.txt", "r")
fileLines = fileRead.readlines()
if fileRead.mode == 'r':
    for readBuffer in fileLines:
        readBuffer = ast.literal_eval(readBuffer)
        allDistanceFingerOpen.append(readBuffer)


dfOpen = pd.DataFrame(allDistanceFingerOpen, columns=['index', 'middle', 'ring','little'])
dfClose = pd.DataFrame(allDistanceFingerClose, columns=['index', 'middle', 'ring','little'])
print("   Open hand: \n" + str(dfOpen.describe()))
print("\n   Close hand: \n" + str(dfClose.describe()))

# plt.hist([dfOpen['index'],dfClose['index']],50,density=True,histtype='bar')


# Draw the density plot
sns.distplot(dfOpen['little'],hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'open')
sns.distplot(dfClose['little'],hist = False, kde = True, kde_kws = {'linewidth': 3}, label = 'close')

# Method 1: on the same Axis
# sns.distplot( dfOpen["index"] , color="skyblue", label="Open")
# sns.distplot( dfClose["index"] , color="red", label="Close")

# plt.hist([dfOpen["index"],dfClose["index"]],10,density=True,histtype='bar',label=["open","close"])

# Plot formatting
plt.title('Open-Close hand distance')

plt.legend()
plt.show()

# capture.release()
cv2.destroyAllWindows()
