from loadData import *
from model import *
from candidates import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json

# Notes:
# Bij de predict, i.p.v. de index van de label op te zoeken op basis van de hoogste waarde ( meerdere classes ), dus softmax,
# wordt nu sigmoid gebruikt. Als de waarde dan dichter bij de label is is het dus wel een kentekenplaat en als die bij de 1 ligt niet.

# Elk plaatje croppen en opslaan als nieuw plaatje (file).
# for i in range(len(loadData()[0])):
#     saveCroppedImage( convertBBoxImage(cutOutBBox(i), (224,224)), 'C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/dataset/Plates/', 'plate_' + loadData()[0][i] + '.jpg') # Image naam moet zelfde naar als originele image bevatten, anders kan bijvoorbeeld plate0 de kentekenplaat van img_2297 bevatten.

# file, xMin, xMax, yMin, yMax, label = loadData()
# for i in range(len(file)):
#     image = np.array(Image.open("C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/images/" + str(file[i]) + ".jpg"))
#     image = cutOutRandom( image, (224, 224), xMin[i], xMax[i], yMin[i], yMax[i] )
#     saveCroppedImage( image, 'C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/dataset/No Plates/', 'no_plate_' + file[i] + '.jpg') # Image naam moet zelfde naar als originele image bevatten, anders kan bijvoorbeeld plate0 de kentekenplaat van img_2297 bevatten.

# classifier = trainModel()

json_file = open('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_arch.json', 'r')
classifier = model_from_json(json_file.read(), custom_objects={'KerasLayer':hub.KerasLayer})
classifier.load_weights('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_weights.h5')

classifier.compile(optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
                loss='binary_crossentropy',
                metrics=['acc'])

license_plate = np.array(Image.open('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Transfer Test/plate.jpg'))/255.0

plateImages = []
noPlateImages = []
for i in range(len(loadData()[0])):
    imagePlate = cv.imread("C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/dataset/Plates/" + "plate_" + str(loadData()[0][i]) + ".jpg")
    imagePlate = cv.cvtColor(imagePlate, cv.COLOR_BGR2RGB)
    imageNoPlate = cv.imread("C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/dataset/No Plates/" + "no_plate_" + str(loadData()[0][i]) + ".jpg")
    imageNoPlate = cv.cvtColor(imageNoPlate, cv.COLOR_BGR2RGB)
    plateImages.append(np.array(imagePlate)/255.0)
    noPlateImages.append(np.array(imageNoPlate)/255.0)

resultPlates = classifier.predict(np.array(plateImages))
print("Image With Plate Predictions: ")
print(resultPlates[:10])

print("Image Without Plate Predictions: ")
resultNoPlates = classifier.predict(np.array(noPlateImages))
print(resultNoPlates[:10])

print("Result Of Single Image: ")
resultSinglePlate = classifier.predict(license_plate[np.newaxis, ...])
print(resultSinglePlate)

classifier.summary()

# license_plate = np.array(Image.open('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Transfer Test/Car-Number-Plate.jpg'))
# license_plate_candidates, bboxes = sweepCandidates( np.array(license_plate), (175, 50))

# results = []
# bboxesResults = []
# for i in range(len(license_plate_candidates)-1):
#     print(license_plate_candidates[i].shape)
#     license_plate_candidate = convertBBoxImage( license_plate_candidates[i], (224, 224) )
#     license_plate_candidate = np.array(license_plate_candidate)/255.0
#     print(license_plate_candidate.shape)
#     results.append( classifier.predict(license_plate_candidate[np.newaxis, ...])[0] )
#     bboxesResults.append( bboxes[i] )

# maxIndex = results.index(max(results))
# print(results[maxIndex])
# if results[maxIndex] > 0.85:
#     cv.rectangle( license_plate, (bboxes[maxIndex][0], bboxes[maxIndex][2]), (bboxes[maxIndex][1], bboxes[maxIndex][3]), (255,0,0), 3 )
# break

# if result[0] > 0.7:
#     cv.rectangle( license_plate, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (255,0,0), 3 )

# print(result.shape)

# file, xMin, xMax, yMin, yMax, label = loadData()
# image = np.array(Image.open("C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/images/" + str(file[0]) + ".jpg"))

# image = cutOutRandom( image, (224, 224), xMin[0], xMax[0], yMin[0], yMax[0] )
# print(image.shape)
# plt.imshow(image)
# plt.show()
plt.imshow(license_plate)
plt.show()