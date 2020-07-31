import cv2
import numpy as np
import random
from src.hand_tracker import HandTracker
from scipy.interpolate import interp1d
from numpy import interp
from scipy.spatial import distance
import glob
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt


WINDOW = "Hand Tracking"
WINDOW2 = "Rectangle Game"
PALM_MODEL_PATH = "models/palm_detection_builtin.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"
IMAGES_PATH = "W:/mockup-handDetection/mockup-handdetection-mediapipe/leapGestRecog/09/01_palm/*"

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



# cv2.namedWindow(WINDOW)
# cv2.namedWindow(WINDOW2)
# capture = cv2.VideoCapture(0)

# if capture.isOpened():
#     hasFrame, frame = capture.read()
# else:
#     hasFrame = False

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
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

# print("Frame shape: "+ str(frame.shape))
counter = 0

# height = frame.shape[0]
# width = frame.shape[1]
# channels = frame.shape[2]

images = glob.glob('W:/mockup-handDetection/mockup-handdetection-mediapipe/leapGestRecog/09/01_palm/*')
images = glob.glob(IMAGES_PATH)
allDistanceFinger = []
for image in images:
    img = cv2.imread(image)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)
    if points is not None:
        # THUMB, INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER = finger_state(points)
        distanceFinger = fingerState_distance_ratio(points)
        # gesture(INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER,display)
        allDistanceFinger.append(distanceFinger)
print(allDistanceFinger)
df = pd.DataFrame(allDistanceFinger, columns=['index', 'middle', 'ring','little'])
boxplot = df.boxplot(column=['index', 'middle', 'ring','little'])
boxplot

fig1, ax1 = plt.subplots()
ax1.boxplot(df)
plt.show()

# while hasFrame:
#     display = np.zeros((height,width,channels),np.uint8)
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     points, _ = detector(image)
#     # if counter==0:
#     #     x,y,w,h = randomRectangle(display)
#     # counter = counter+1
#     if points is not None:
#         newPoints = []
#         for point in points:
#             x, y = point
#             cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
#             # cv2.putText(frame, str(int(x))+","+str(int(y)), (int(x)+4, int(y)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
#         # print(str(distance.euclidean(points[5], points[8]))+" , "+str(distance.euclidean(points[0], points[8])))

#         # mapping new xy point
#         for point in points:
#             x, y = point
#             newPoint = []
#             newX = mappingPointX(frame,display,x)
#             newY = mappingPointY(frame,display,y)
#             newPoint.append(newX)
#             newPoint.append(newY)
#             newPoints.append(newPoint)
#         # print("Old point: " + str(points))
#         # print("New point: " + str(newPoints))

#         for point in newPoints:
#             x, y = point
#             cv2.circle(display, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)

#         # track the first finger points[8]
#         xP, yP = newPoints[0]
#         # cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
#         # cv2.putText(frame, str(int(x))+","+str(int(y)), (int(x)+4, int(y)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
#         cv2.circle(display, (int(xP), int(yP)), THICKNESS * 2, POINT_COLOR, THICKNESS)
#         cv2.putText(display, str(int(xP))+","+str(int(yP)), (int(xP)+4, int(yP)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

#         # cv2.rectangle(display,(90,90),(200,200),(255,0,0), 1, 8)
#         # isPointIn = isHandInRectngle(xP,yP,90,90,200,200)
#         # if isPointIn:
#         #     cv2.putText(display, "Innnnnnnnnnnnnnnnnnn", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

#         # THUMB, INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER = finger_state(points)
#         INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER, distanceFinger = fingerState_distance_ratio(points)
#         gesture(INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER,display)
#         allDistanceFinger.append(distanceFinger)

#         # connection camera screen
#         for connection in connections:
#             x0, y0 = points[connection[0]]
#             x1, y1 = points[connection[1]]
#             cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

#         # connection display screen
#         for connection in connections:
#             x0, y0 = newPoints[connection[0]]
#             x1, y1 = newPoints[connection[1]]
#             cv2.line(display, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

#     cv2.imshow(WINDOW, frame)
#     hasFrame, frame = capture.read()
#     cv2.imshow(WINDOW2,display)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# capture.release()
cv2.destroyAllWindows()
