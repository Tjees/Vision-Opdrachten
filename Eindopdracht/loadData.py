import pandas as pd
import cv2 as cv
import numpy as np
from PIL import Image
import random

# Op internet opgezocht:
# Data laden vanuit CSV file met pandas.
# Border maken met openCV.
# Waarom list soms handiger is met appenden dan numpy array.
# Beter manier om te croppen.
# Voorbeeld van resizen met behouden van aspect ratio ( als width of height oneven is is aspect ratio niet precies behouden maar zo goed mogelijk.)

# Notes:
# Bij het maken van de grijze borders moet je aan beide kanten een border toevoegen,
# Hierdoor kan je dus bij een oneven getal, bijvoorbeeld 153, een border size van 153/2 = 76,5 krijgen wat je dus moet afronden.
# Hierdoor krijg je dus een width of height pixel te weinig en moet je die dus later toevoegen dus kan het does niet precies
# in het midden omdat je niet aan beide kanten een halve pixel toe kan voegen.
# Ongeveer hetzelfde geval bij het resizen maar daar kan je gewoon de size meegeven i.p.v. de shape * ratio en dan doet hij het automatisch,
# de aspect ratio wordt dus waarschijnlijk niet precies behouden en zal dus met ongeveer een pixel afwijken. Maar maakt waarschijnlijk niet heel veel uit.
# Nu pak ik alleen de corners als image die geen kentekenplaat bevat maar misschien is het handiger om een functie te maken die een random 224x224 image pakt
# uit de originele image die niet binnen de bounding box valt.
# Het beste is waarschijnlijk om een functie te maken die per plaatje op een random plek 224x224 kiest en die returned. Dan heb je
# net zo veel plaatjes waar geen kentekens op staan dan waar wel kentekens op staan en het zijn random plekken dus niet steeds dezelfde. ( Heb nu ongeveer 300 waar wel iets op staat en 1200 waar niets op staan. ).

# Laadt de data in vanuit CSV file.
def loadData():
    df = pd.read_csv('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/annotation_car_plate_train.csv')
    file = df['file'].to_numpy()
    xMin = df['xmin'].to_numpy()
    xMax = df['xmax'].to_numpy()
    yMin = df['ymin'].to_numpy()
    yMax = df['ymax'].to_numpy()
    label = df['name'].to_numpy()
    return file, xMin, xMax, yMin, yMax, label

# Knipt de bounding box uit het plaatje.
def cutOutBBox( i ):
    file, xMin, xMax, yMin, yMax, label = loadData()
    file, xMin, xMax, yMin, yMax, label = file[i], xMin[i], xMax[i], yMin[i], yMax[i], label[i]

    path = "C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/images/" + str(file) + '.jpg'
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    croppedImage = image[yMin:yMax,xMin:xMax]

    return np.array(croppedImage)

# Resized de bounding box en behoudt de aspect ratio ( dus het hele bounding box plaatje ).
def resizeBBoxImage( img, size ):
    newImage = img
    if( newImage.shape[0] > size[1] ):
        ratio = size[1] / newImage.shape[0]
        newImage = cv.resize(newImage,(int(newImage.shape[1] * ratio), size[1]))
    if( newImage.shape[1] > size[0] ):
        ratio = size[0] / newImage.shape[1]
        newImage = cv.resize(newImage,(size[0], int(newImage.shape[0] * ratio)))

    return np.array(newImage)

# Resized de bounding box maar behoudt detail en kapt dus een deel van het bounding box plaatje af.
def cropBBoxImage( img, size ):
    newImage = img
    if( newImage.shape[0] > size[1] ):
        newImage = newImage[ int((newImage.shape[0]-size[1])/2) : int(img.shape[0]-((newImage.shape[0]-size[1])/2)) ]
    if( newImage.shape[1] > size[0] ):
        newImage = newImage[ :, int((newImage.shape[1]-size[0])/2) : int(img.shape[1]-((newImage.shape[1]-size[0])/2))]
    
    return np.array(newImage)

# Voegt grijze borders toe aan de image van gelijke grote ( dus halve pixels worden afgerond naar beneden ).
def borderBBoxImage( img, size ):
    newImage = img
    borderColor = (125, 125, 125)
    if( newImage.shape[0] < size[1] ):
        borderY = int((size[1]-newImage.shape[0])/2)
        newImage = cv.copyMakeBorder(newImage,borderY,borderY,0,0,cv.BORDER_CONSTANT,value=borderColor)

    if( newImage.shape[1] < size[0] ):
        borderX = int((size[0]-newImage.shape[1])/2)
        newImage = cv.copyMakeBorder(newImage,0,0,borderX,borderX,cv.BORDER_CONSTANT,value=borderColor)

    return np.array(newImage)

# Convert de bbox image ( resized en voegt borders toe aan alle kanten ) en checkt of de image wel voldoet aan de size
# en voegt anders nog een pixel grote border toe ( dus die halve pixels die niet aan beide kanten konden worden neergezet
# worden bij elkaar opgeteld en aan een kant toegevoegd ).
def convertBBoxImage( img, size ):
    convertedImage = borderBBoxImage( resizeBBoxImage( img, size ), size )
    borderColor = (125, 125, 125)
    if( convertedImage.shape[0] < size[1] ):
        convertedImage = cv.copyMakeBorder(convertedImage,0,1,0,0,cv.BORDER_CONSTANT,value=borderColor)
    if( convertedImage.shape[1] < size[0] ):
        convertedImage = cv.copyMakeBorder(convertedImage,0,0,0,1,cv.BORDER_CONSTANT,value=borderColor)

    return np.array(convertedImage)

# Knipt de hoeken uit van de image, te gebruiken met de originele image om backgrounds te maken maar alleen handig als het object
# rond het midden zit. Is wel makkelijker in sommige gevallen. Ook is het sneller ( want je hoeft niet voor elk stukje te checken of de bounding box erin zit ).
def cutOutCorners( img, size ):
    corners = []
    topLeftCorner = img[0:size[1], 0:size[0]]
    topRightCorner = img[0:size[1], img.shape[1]-size[0]:img.shape[1]]
    bottomLeftCorner = img[img.shape[0]-size[1]:img.shape[0], 0:size[0]]
    bottomRightCorner = img[img.shape[0]-size[1]:img.shape[0], img.shape[1]-size[0]:img.shape[1]]
    corners.append( topLeftCorner )
    corners.append( topRightCorner )
    corners.append( bottomLeftCorner )
    corners.append( bottomRightCorner )
    return np.array(corners)

# Checken of de random image de bbox image bevat.
def checkBBoxIntersect( bboxX, bboxY, randomX, randomY):
    for y in range(randomY[0], randomY[1]):
        for x in range( randomX[0], randomX[1] ):
            if( (x == bboxX[0] and y == bboxY[0]) or (x == bboxX[1] and y == bboxY[1]) ):
                return True
    return False

# Random stukje van de image uitknippen om als background te gebruiken, handiger dan de corners want checkt of het stukje niet in de
# bounding box zit en is ook random dus is handiger in sommige gevallen.
def cutOutRandom( img, size, bboxXMin, bboxXMax, bboxYMin, bboxYMax ):
    randomXMin = random.randint(0, img.shape[1]-size[0])
    randomYMin = random.randint(0, img.shape[0]-size[1])
    randomXMax = randomXMin + size[0]
    randomYMax = randomYMin + size[1]

    while checkBBoxIntersect( (bboxXMin, bboxXMax), (bboxYMin, bboxYMax), (randomXMin, randomXMax), (randomYMin, randomYMax)):
        randomXMin = random.randint(0, img.shape[1]-size[0])
        randomYMin = random.randint(0, img.shape[0]-size[1])
        randomXMax = randomXMin + size[0]
        randomYMax = randomYMin + size[1]

    newImage = img[randomYMin:randomYMax, randomXMin:randomXMax]
    return np.array(newImage)

# Laadt alle converted images in de dataset met bijbehorende labels. Dus maakt een nieuwe dataset met converted images.
def loadDataSet():
    file, xMin, xMax, yMin, yMax, label = loadData()
    data, labels = [], []
    for i in range(len(loadData()[0])):
        data.append(convertBBoxImage(cutOutBBox(i), (224,224)))
        if( loadData()[5][i] == "plate" ):
            labels.append(1)
        else:
            labels.append(0)
    for i in range(len(loadData()[0])):
        image = cv.imread("C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/images/" + str(file[i]) + ".jpg")
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cutOutRandom( image, (224, 224), xMin[i], xMax[i], yMin[i], yMax[i] )
        data.append( image )
        labels.append(0)

    return np.array(data), np.array(labels)

# Functie om afbeeldingen op te slaan als file.
def saveCroppedImage( croppedImage, filePath, imageName ):
    im = Image.fromarray(croppedImage)
    im.save(filePath + '/' + imageName)