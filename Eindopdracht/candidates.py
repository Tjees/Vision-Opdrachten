from loadData import *
import cv2 as cv
import numpy as np
from PIL import Image
import random

def randomCandidate( img, size ):
    xMin = random.randint(0,img.shape[1]-size[0])
    xMax = xMin + size[0]
    yMin = random.randint(0,img.shape[0]-size[1])
    yMax = yMin + size[1]
    img = img[yMin:yMax, xMin:xMax]
    return img, (xMin, xMax, yMin, yMax)

def testCandidate( img ):
    img = img[270:335, 230:415]
    return img, (230, 415, 270, 335)

def sweepCandidates( img, size ):
    candidates = []
    bboxes = []
    posX = 0
    posY = 0
    while posY < ( img.shape[0] - size[1] ):
        while posX < ( img.shape[1] - size[0] ):
            candidates.append(img[posY:posY+size[1], posX:posX+size[0]])
            bboxes.append((posX, posX+size[0], posY, posY+size[1]))
            posX += size[0]//4
        posY += size[1]//4
        posX = 0
    
    return np.array(candidates), np.array(bboxes)