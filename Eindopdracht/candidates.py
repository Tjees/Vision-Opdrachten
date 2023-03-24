from loadData import *
import cv2 as cv
import numpy as np
from PIL import Image
import random
import math

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

def createHoughLines( img ):
    gray_image = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    clustered_image = createCandidatePositions(createClustering(img))
    edges = cv.Canny(clustered_image,400,425,apertureSize = 3)

    lines = cv.HoughLines(edges,1,np.pi/180.0,125)
    maxLineLengthY = img.shape[0]
    maxLineLengthX = img.shape[1]
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + maxLineLengthX*(-b))
            y1 = int(y0 + maxLineLengthX*(a))
            x2 = int(x0 - maxLineLengthX*(-b))
            y2 = int(y0 - maxLineLengthX*(a))

            #cv.circle(img, (int(x0), int(y0)), 3, (255,0,0),2)
            cv.line(img,(x1,y1),(x2,y2),(255,0,0),2)

    # Vertical lines
    # lines = cv.HoughLinesP(
    #     edges, 1, np.pi, threshold=100, minLineLength=100, maxLineGap=10)

    # lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=50,maxLineGap=100)
    # for line in lines:
    #     x1,y1,x2,y2 = line[0]
    #     cv.line(img,(x1,y1),(x2,y2),(255,0,0),2)

    return img

def createClustering( img ):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 10
    ret,labels,centers=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    print(centers)
    print(labels)
    print(centers[labels[0][0]])
    # centers[0] = [0,0,0]
    # centers[1] = [255,255,255]

    # thresWhite = list(centers[3])
    thresWhite = [100,100,100]
    thresYellow = [150,150,50]
    for i in range(len(centers)):
        center = list(centers[i])
        if center[0] > thresWhite[0] and center[1] > thresWhite[1] and center[2] > thresWhite[2]:
            centers[i] = [255,255,255]
        elif center[0] > thresYellow[0] and center[1] > thresYellow[1] and center[2] < thresYellow[2]:
            centers[i] = [255,255,0]
        else:
            centers[i] = [0,0,0]

    res = centers[labels.flatten()]
    res = res.reshape((img.shape))
    
    return res

def createCandidatePositions(img):
    candidates = []
    bboxes = []
    
    clusteredImage = createClustering(img)

    # convert image to grayscale image
    gray_image = cv.cvtColor(clusteredImage, cv.COLOR_RGB2GRAY)

    # for i in range(10):
    #     cv.GaussianBlur(gray_image,(5,5),0)

    # kernelClose = np.ones((50,50),np.uint8)
    # kernelErode = np.ones((20,20),np.uint8)
    # kernelClose = np.ones((img.shape[1]//500,img.shape[0]//500),np.uint8)
    # kernelErode = np.ones((img.shape[1]//300,img.shape[0]//300),np.uint8)
    kernelClose = np.ones((int(img.shape[1]*0.0125),int(img.shape[0]*0.015)),np.uint8)
    kernelErode = np.ones((int(img.shape[1]*0.005),int(img.shape[0]*0.0065)),np.uint8)
    closing = cv.morphologyEx(gray_image, cv.MORPH_CLOSE, kernelClose)
    closing = cv.morphologyEx(closing, cv.MORPH_ERODE, kernelErode)
    # closing = createClustering(closing)

    edges = cv.Canny(closing,400,425,apertureSize = 3)

    # calculate moments of binary image
    # find contours in the binary image
    contours, hierarchy = cv.findContours(closing,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img, contours, -1, (255,0,0), 3)
    for c in contours:
        # calculate moments for each contour
        M = cv.moments(c)
        area = cv.contourArea(c)
        
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # cv.rectangle( img, (cX,cY), (cX+10, cY+10), (255,0,0), 3 )
            # size = (448,224)
            size = (int(img.shape[1]*0.115),int(img.shape[0]*0.075))
            xMin = cX - size[0]
            xMax = cX + size[0]
            yMin = cY - size[1]
            yMax = cY + size[1]

            # cv.rectangle( img, (xMin+(size[0]//4), yMin+(size[0]//4)), (xMax-(size[0]//4), yMax-(size[0]//4)), (255,0,0), 3 )
            # cv.rectangle( img, (xMin, yMin), (xMax, yMax), (255,0,0), 3 )

            if( yMin > 0 and yMax > 0 and xMin > 0 and yMax > 0 ):
                candidates.append(img[yMin:yMax, xMin:xMax])
                bboxes.append((xMin, xMax, yMin, yMax))
                candidates.append(img[yMin+(size[0]//4):yMax-(size[0]//4), xMin+(size[0]//4):xMax-(size[0]//4)])
                bboxes.append((xMin+(size[0]//4), xMax-(size[0]//4), yMin+(size[0]//4), yMax-(size[0]//4)))
            # cv.rectangle( img, (xMin+(size[0]//2), yMin+(size[0]//2)), (xMax-(size[0]//2), yMax-(size[0]//2)), (255,0,0), 3 )
        else:
            cX, cY = 0, 0

    return np.array(candidates), np.array(bboxes)
    # return img
    # return cv.cvtColor(closing, cv.COLOR_GRAY2RGB)