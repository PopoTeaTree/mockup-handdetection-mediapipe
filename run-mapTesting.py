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

xCroped= mappingPointX(10,5,6)
yCroped = mappingPointY(10,5,6)
print("mapping point crop frame: " + str(xCroped)+","+str(yCroped))