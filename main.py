import numpy
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
from skimage import data
from skimage.viewer import ImageViewer
from skimage import io
from skimage import color

def setColorRange( image ):
    hsvImage = rgb2hsv( image )
    for i in hsvImage:
        for j in i:
            if j[0] > ( 15 / 360 ) and j[0] < ( 345 / 360 ) : # Waarde in graden delen door 360 om getal tussen 0 en 1 te krijgen.
                # if j[0] < ( 15 / 360 ) or j[0] > ( 345 / 360 ) voor grijs maken telefoon hokje en achtergrond behouden.
                # if j[0] > ( 15 / 360 ) and j[0] < ( 345 / 360 ) voor grijs maken achtergrond en telefoon hokje behouden.
                j[1] = 0
    return hsv2rgb(hsvImage) # Moet terug naar RGB want de viewer gebruikt geen HSV maar RGB waardes.

def makeHueArray( image ):
    hueArray = []
    for i in image:
        for j in i:
            hueArray.append( j[0] * 360 ) # Hue value is 0 - 1 dus keer 360 want dan krijg je de hue.
    return hueArray


image = io.imread('C:/Users/tjezv/OneDrive/Afbeeldingen/London.jpg')
setColorRange( image )

fig, axs = plt.subplots( 2, 3, figsize=(8,4) )

axs[0, 0].imshow( image )
axs[0, 1].imshow( setColorRange( image ) )

axs[1, 0].hist( makeHueArray( rgb2hsv( image ) ) )
axs[1, 1].hist( makeHueArray( rgb2hsv( setColorRange( image ) ) ) )

plt.show()

# viewer = ImageViewer(setColorRange(image))
# viewer.show()