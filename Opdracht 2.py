import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.viewer import ImageViewer
from skimage import io
from skimage import color
from skimage import data, filters
from scipy import ndimage

# Aparte imports voor functies weggehaald want is niet echt nodig.

def makeHueArray( image ):
    hueArray = []
    for i in image:
        for j in i:
            hueArray.append( j[0] * 360 ) # Hue value is 0 - 1 dus keer 360 want dan krijg je de hue.
    return hueArray

image = io.imread('C:/Users/tjezv/OneDrive/Afbeeldingen/London.jpg') # RGB test afbeelding of image.
print( image )

grayimage = color.rgb2gray( image ) # Filter werkt momenteel alleen met grayscale images.
print( grayimage )

mask1=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]
# mask1 = np.array([[1,1,1],[1,1,0],[1,0,0]])
newimage=ndimage.convolve(grayimage, mask1)
newimage=ndimage.convolve(newimage, mask1)

fig, axs = plt.subplots( 2, 3, figsize=(8,4) )

axs[0, 0].imshow( image )
axs[0, 1].imshow( color.gray2rgb( newimage ) ) # Terug naar rgb image anders werkt het displayen niet.

axs[1, 0].hist( makeHueArray( color.rgb2hsv( image ) ) )
axs[1, 1].hist( makeHueArray( color.rgb2hsv( color.gray2rgb( newimage ) ) ) )

plt.show()