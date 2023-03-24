import pandas as pd
import cv2 as cv
import numpy as np
from PIL import Image

# Op internet opgezocht:
# Data laden vanuit CSV file met pandas.
# Border maken met openCV.
# Waarom list soms handiger is met appenden dan numpy array.
# Beter manier om te croppen.

def loadData():
    # load the CSV file into a DataFrame
    df = pd.read_csv('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/annotation_car_plate_train.csv')
    file = df['file'].to_numpy() #Kan een list maken van de kolom maar is misschien minder flexibel. #file = list(df['file'])
    xMin = df['xmin'].to_numpy()
    xMax = df['xmax'].to_numpy()
    yMin = df['ymin'].to_numpy()
    yMax = df['ymax'].to_numpy()
    return file, xMin, xMax, yMin, yMax

def cutOutBBox( i ):
    #Load image data.
    file, xMin, xMax, yMin, yMax = loadData()
    file, xMin, xMax, yMin, yMax = file[i], xMin[i], xMax[i], yMin[i], yMax[i]

    #Load actual image.
    path = 'C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/images/' + str(file) + '.jpg'
    # io.imread() doet het goed en cv.imread() verwisseld kleuren. Cv2 gebruikt geen RGB als default maar BGR.
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Dus na het inladen eerst van BGR naar RGB converteren.

    #Crop image in y direction.
    #yCrop = image[yMin:yMax]
    croppedImage = image[yMin:yMax,xMin:xMax]

    # Crop yCrop image in x direction.
    # xCrop = [] # Normale array omdat numpy array fixed size is dus is vervelend (moet eerst size bepalen).
    # for i in range(len(yCrop)):
    #     xCrop.append(yCrop[i][xMin:xMax])
    #xCrop = yCrop[:,xMin:xMax]

    # Return final cropped image.
    return np.array(croppedImage) # Final image kan wel numpy array zijn want is fixed size en wordt ook niet later aangepast.

def resizeBBoxImage( img, size ):
    newImage = img
    if( newImage.shape[0] > size[1] ):
        middle = (newImage.shape[0]-1)/2 # Around middle haha.
        newImage = newImage[ int(middle - (size[1]/2)) : int(middle + (size[1]/2))]

    if( newImage.shape[1] > size[0] ):
        middle = (newImage.shape[1]-1)/2 # Around middle haha.
        # Misschien nog aanpassen/vervangen met andere implementatie.
        newImage = newImage[: , int(middle - (size[0]/2)) : int(middle + (size[0]/2))] # In dit geval wel handiger/makkelijker haha.

    return np.array(newImage)

def borderBBoxImage( img, size ):
    newImage = img
    borderColor = (125, 125, 125)
    if( newImage.shape[0] < size[1] ):
        borderY = int((size[1]-newImage.shape[0])/2) # Border voor top en bottom.
        newImage = cv.copyMakeBorder(newImage,borderY,borderY,0,0,cv.BORDER_CONSTANT,value=borderColor)

    if( newImage.shape[1] < size[0] ):
        borderX = int((size[0]-newImage.shape[1])/2) # Border voor links en rechts.
        newImage = cv.copyMakeBorder(newImage,0,0,borderX,borderX,cv.BORDER_CONSTANT,value=borderColor)

    return np.array(newImage) # constantBorderImage.

def convertBBoxImage( img, size ):
    convertedImage = borderBBoxImage( resizeBBoxImage( img, size ), size )
    borderColor = (125, 125, 125)
    if( convertedImage.shape[0] < size[1] ): # Shape is altijd 244 of kleiner (243) want rond af naar beneden en niet naar boven.
        convertedImage = cv.copyMakeBorder(convertedImage,0,1,0,0,cv.BORDER_CONSTANT,value=borderColor) # Rij toevoegen aan bottom.
    if( convertedImage.shape[1] < size[0] ): # Shape is altijd 244 of kleiner (243) want rond af naar beneden en niet naar boven.
        convertedImage = cv.copyMakeBorder(convertedImage,0,0,0,1,cv.BORDER_CONSTANT,value=borderColor) # Rij toevoegen aan rechts.

    return np.array(convertedImage)

def saveCroppedImage( croppedImage, filePath, imageName ):
    im = Image.fromarray(croppedImage)
    im.save(filePath + '/' + imageName)