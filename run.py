import cv2
import numpy as np
import random
from src.hand_tracker import HandTracker

WINDOW = "Hand Tracking"
WINDOW2 = "Rectangle Game"
PALM_MODEL_PATH = "models/palm_detection_builtin.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

# def gesture(THUMB, INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER,frame):
#     if not THUMB and INDEXFINGER and not MIDDLEFINGER and not RINGFINGER and not LITTLEFINGER:
#         cv2.putText(frame, "Pointing", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
#     elif THUMB and INDEXFINGER and not MIDDLEFINGER and not RINGFINGER and not LITTLEFINGER:
#         cv2.putText(frame, "Gun", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

# def finger_state(points,frame):
#     # finger state
#     THUMB = False
#     INDEXFINGER = False
#     MIDDLEFINGER = False
#     RINGFINGER = False
#     LITTLEFINGER = False

#     pseudoFixKeyPoint1 = points[2][0]
#     if points[3][0] < pseudoFixKeyPoint1 and points[4][0] < pseudoFixKeyPoint1:
#         THUMB = True
#         print("Trumb UP")
#         # cv2.putText(frame, "Trumb UP", (points[2][0], points[2][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

#     pseudoFixKeyPoint2 = points[6][1]
#     if points[7][1] < pseudoFixKeyPoint2 and points[8][1] < pseudoFixKeyPoint2:
#         INDEXFINGER = True
#         print("INDEXFINGER")
#         # cv2.putText(frame, "INDEXFINGER", (points[6][0], points[6][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

#     pseudoFixKeyPoint3 = points[10][1]
#     if points[11][1] < pseudoFixKeyPoint3 and points[12][1] < pseudoFixKeyPoint3:
#         MIDDLEFINGER = True
#         print("MIDDLEFINGER")
#         # cv2.putText(frame, "MIDDLEFINGER", (points[10][0], points[10][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

#     pseudoFixKeyPoint4 = points[14][1]

#     if points[15][1] < pseudoFixKeyPoint4 and points[16][1] < pseudoFixKeyPoint4:
#         RINGFINGER = True
#         print("RINGFINGER")
#         # cv2.putText(frame, "RINGFINGER", (points[14][0], points[14][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

#     pseudoFixKeyPoint5 = points[18][1]
#     if points[19][1] < pseudoFixKeyPoint5 and points[20][1] < pseudoFixKeyPoint5:
#         LITTLEFINGER = True
#         print("LITTLEFINGER")
#         # cv2.putText(frame, "LITTLEFINGER", (points[20][0], points[20][1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
#     return THUMB, INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER
def xyrandom():
    return x = random.randint(0, 380), y = random.randint(0, 540)

cv2.namedWindow(WINDOW)
cv2.namedWindow(WINDOW2)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

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

print("Frame shape: "+ str(frame.shape))

while hasFrame:
    display = np.zeros((480,640,3),np.uint8)
    # cv2.rectangle( display,(x, y) ,(x+100,y+100),(255,0,0), 1, 8)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)
    if points is not None:
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            # cv2.putText(frame, str(int(x))+","+str(int(y)), (int(x)+4, int(y)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        for point in points:
            x, y = point
            cv2.circle(display, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        # track the first finger points[8]
        # x, y = points[8]
        # cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        # cv2.putText(frame, str(int(x))+","+str(int(y)), (int(x)+4, int(y)+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # THUMB, INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER = finger_state(points,frame)
        # gesture(THUMB, INDEXFINGER, MIDDLEFINGER, RINGFINGER, LITTLEFINGER,frame)

        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(display, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    cv2.imshow(WINDOW2,display)

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
