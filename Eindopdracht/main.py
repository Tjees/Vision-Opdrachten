from loadData import *
import matplotlib.pyplot as plt
import numpy as np

# Elk plaatje croppen en opslaan als nieuw plaatje (file).
# for i in range(len(loadData()[0])):
#     saveCroppedImage( cutOutBBox(i), 'C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/nummerborden', 'plate_' + loadData()[0][i] + '.jpg') # Image naam moet zelfde naar als originele image bevatten, anders kan bijvoorbeeld plate0 de kentekenplaat van img_2297 bevatten.

kentekenplaat = cutOutBBox(0)
image = convertBBoxImage(kentekenplaat, (244,244))
print(image.shape)
plt.imshow(image)
plt.show()