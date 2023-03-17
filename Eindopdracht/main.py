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
# for i in range(len(loadData()[0])):
#     image = np.array(Image.open("C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/images/" + str(file[i]) + ".jpg"))
#     image = cutOutRandom( image, (224, 224), xMin[i], xMax[i], yMin[i], yMax[i] )
#     saveCroppedImage( image, 'C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/dataset/No Plates/', 'no_plate_' + loadData()[0][i] + '.jpg') # Image naam moet zelfde naar als originele image bevatten, anders kan bijvoorbeeld plate0 de kentekenplaat van img_2297 bevatten.

json_file = open('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_arch.json', 'r')
classifier = model_from_json(json_file.read(), custom_objects={'KerasLayer':hub.KerasLayer})
classifier.load_weights('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_weights.h5')

classifier.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

# classifier = trainModel()

license_plate = np.array(Image.open('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Transfer Test/car.jpg'))
license_plate_candidates, bboxes = sweepCandidates( np.array(license_plate), (320, 240))

for i in range(len(license_plate_candidates)-1):
    print(license_plate_candidates[i].shape)
    license_plate_candidate = convertBBoxImage( license_plate_candidates[i], (224, 224) )
    license_plate_candidate = np.array(license_plate_candidate)/255.0
    print(license_plate_candidate.shape)
    if classifier.predict(license_plate_candidate[np.newaxis, ...]) > 0.85:
        cv.rectangle( license_plate, (bboxes[i][0], bboxes[i][2]), (bboxes[i][1], bboxes[i][3]), (255,0,0), 3 )
        # break

# result = classifier.predict(license_plate_candidate[np.newaxis, ...])
# if result[0] > 0.7:
#     cv.rectangle( license_plate, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (255,0,0), 3 )

# print(result)
# print(result.shape)

classifier.summary()

# file, xMin, xMax, yMin, yMax, label = loadData()
# image = np.array(Image.open("C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/images/" + str(file[0]) + ".jpg"))

# image = cutOutRandom( image, (224, 224), xMin[0], xMax[0], yMin[0], yMax[0] )
# print(image.shape)
# plt.imshow(image)
# plt.show()
plt.imshow(license_plate)
plt.show()