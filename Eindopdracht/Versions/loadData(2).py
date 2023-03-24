import pandas as pd
from skimage import io
import numpy as np
from PIL import Image

def loadData():
    # load the CSV file into a DataFrame
    df = pd.read_csv('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/annotation_car_plate_train.csv')
    file = df['file'].to_numpy() # Kan een list maken van de kolom maar is misschien minder flexibel of snel. #file = list(df['file'])
    xMin = df['xmin'].to_numpy()
    xMax = df['xmax'].to_numpy()
    yMin = df['ymin'].to_numpy()
    yMax = df['ymax'].to_numpy()
    return file, xMin, xMax, yMin, yMax

def cutOutBBox( index ):
    # Load image data.
    file, xMin, xMax, yMin, yMax = loadData()
    file, xMin, xMax, yMin, yMax = file[index], xMin[index], xMax[index], yMin[index], yMax[index]

    # Load actual image.
    path = 'C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/images/' + str(file) + '.jpg'
    image = io.imread(path)

    # Crop image in y direction.
    yCrop = image[yMin:yMax]
    print(type(yCrop))

    # Crop yCrop image in x direction.
    xCrop = [] # Normale array omdat numpy array fixed size en shape is dus is vervelend (moet eerst size bepalen).
    # De size moet vooraf al bekend zijn. (Voordat je append.)
    # Kan dus misschien wel maar moet dan bijvoorbeeld een numpy.empty((yMax - yMin, xMax - xMin)) maken met dus de size (yMax - yMin, xMax - xMin) en die dan vullen.
    # Numpy append is dus anders dan normale append.
    # Zie in test.py.
    for i in range(len(yCrop)):
        xCrop.append(yCrop[i][xMin:xMax])
    
    #xCrop = yCrop[:,xMin:xMax] (Dit is sneller maar wat er nu staat is langzamer maar hoeft maar een keer dus kan ook gewoon ( Maakt dus niet veel uit in dit geval )).

    # Return final cropped image.
    return xCrop # Final image kan wel numpy array zijn want is fixed size en shape en wordt ook niet later aangepast.
    #return np.array(xCrop)
    #return np.array(xCrop), xCrop

#def convertCroppedImage( image ):

def saveCroppedImage( croppedImage, filePath, imageName ):
    croppedImage = np.array(croppedImage)
    im = Image.fromarray(croppedImage) # Numpy array meegeven want normale array (list) mag niet. (Tuple misschien wel?)
    im.save(filePath + '/' + imageName)