import numpy
import matplotlib
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
from skimage import data
from skimage.viewer import ImageViewer
from skimage import io

def setColorRange( image ):
    hsvImage = rgb2hsv( image )
    for i in hsvImage:
        for j in i:
            if j[0] > ( 15 / 360 ) and j[0] < ( 345 / 360 ) : # Waarde in graden delen door 360 om getal tussen 0 en 1 te krijgen.
                # if j[0] < ( 15 / 360 ) or j[0] > ( 345 / 360 ) voor grijs maken telefoon hokje en achtergrond behouden.
                # if j[0] > ( 15 / 360 ) and j[0] < ( 345 / 360 ) voor grijs maken achtergrond en telefoon hokje behouden.
                j[1] = 0
    return hsv2rgb(hsvImage) # Moet terug naar RGB want de viewer gebruikt geen HSV maar RGB waardes.


image = io.imread('C:/Users/tjezv/OneDrive/Afbeeldingen/London.jpg')
setColorRange( image )

viewer = ImageViewer(setColorRange(image))
viewer.show()