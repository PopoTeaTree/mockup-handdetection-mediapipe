import cv2
import cv

class GestureGuide():
    def __init__(points):
        # finger state
        THUMB = False
        INDEXFINGER = False
        MIDDLEFINGER = False
        RINGFINGER = False
        LITTLEFINGER = False
