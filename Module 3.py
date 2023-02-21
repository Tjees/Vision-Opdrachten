import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.viewer import ImageViewer
from skimage import io
from skimage import color
from skimage import data, filters
from scipy import ndimage
import math
from skimage.util import random_noise
from skimage import feature
# Aparte imports voor functies weggehaald want is niet echt nodig.

smooth=[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]

prewittX=[[-1,0,1],[-1,0,1],[-1,0,1]]
prewittY=[[-1,-1,-1],[0,0,0],[1,1,1]]

sobelX=[[-1,0,1],[-2,0,2],[-1,0,1]]
sobelY=[[-1,-2,-1],[0,0,0],[1,2,1]]

robertsX=[[1,0],[0,-1]]
robertsY=[[0,1],[-1,0 ]]

testX=[[-1,0,1]]
testY=[[-1],[0],[1]]

image = io.imread('C:/Users/tjezv/OneDrive/Afbeeldingen/Coin.jpg') # RGB test afbeelding of image.

grayimage = color.rgb2gray( image ) # Filter werkt momenteel alleen met grayscale images.

newImageSmooth=ndimage.convolve(grayimage, smooth)
newImageSmooth=ndimage.convolve(newImageSmooth, smooth)

prewittGx=ndimage.convolve(grayimage, prewittX)
prewittGy=ndimage.convolve(grayimage, prewittY)
resultPrewitt = np.sqrt( prewittGx ** 2 + prewittGy ** 2 )

sobelGx=ndimage.convolve(grayimage, sobelX)
sobelGy=ndimage.convolve(grayimage, sobelY)
resultSobel = np.sqrt( sobelGx ** 2 + sobelGy ** 2 )

robertsGx=ndimage.convolve(grayimage, robertsX)
robertsGy=ndimage.convolve(grayimage, robertsY)
resultRoberts = np.sqrt( robertsGx ** 2 + robertsGy ** 2 )

testGx=ndimage.convolve(grayimage, robertsX)
testGy=ndimage.convolve(grayimage, robertsY)
resultTest = np.add( testGx, testGy )

edge_prewitt = filters.prewitt(grayimage)
edge_sobel = filters.sobel(grayimage)
edge_roberts = filters.roberts(grayimage)

imageCanny = ndimage.gaussian_filter(grayimage, 4)
imageCanny = random_noise(imageCanny, mode='speckle', mean=0.1)

# Compute the Canny filter for two values of sigma
edge_canny1 = feature.canny(imageCanny)
edge_canny2 = feature.canny(imageCanny, sigma=1.6)

fig, axs = plt.subplots( 4, 3, figsize=(8,4) )

axs[0, 0].imshow( image )
axs[0, 0].set_title( 'Original Image', fontsize=10 )

axs[0, 1].imshow( newImageSmooth, cmap='gray' ) # Terug naar rgb image anders werkt het displayen niet.
axs[0, 1].set_title( 'Gray Image', fontsize=10 )

axs[0, 2].imshow( resultTest, cmap='gray' )

axs[1, 0].imshow( resultPrewitt, cmap='gray' )
axs[1, 1].imshow( resultSobel, cmap='gray' )
axs[1, 2].imshow( resultRoberts, cmap='gray' )

axs[2, 0].imshow( edge_prewitt, cmap='gray' )
axs[2, 1].imshow( edge_sobel, cmap='gray' )
axs[2, 2].imshow( edge_roberts, cmap='gray' )

axs[3, 0].imshow( imageCanny, cmap='gray' )
axs[3, 1].imshow( edge_canny1, cmap='gray' )
axs[3, 2].imshow( edge_canny2, cmap='gray' )

plt.show()
# for row in resultTest:
#     for pixel in row:
#         print(pixel)