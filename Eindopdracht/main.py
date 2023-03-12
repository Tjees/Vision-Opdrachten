from loadData import *
from model import *
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
#     saveCroppedImage( convertBBoxImage(cutOutBBox(i), (224,224)), 'C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/dataset', 'plate_' + loadData()[0][i] + '.jpg') # Image naam moet zelfde naar als originele image bevatten, anders kan bijvoorbeeld plate0 de kentekenplaat van img_2297 bevatten.

json_file = open('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_arch.json', 'r')
classifier = model_from_json(json_file.read(), custom_objects={'KerasLayer':hub.KerasLayer})
classifier.load_weights('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_weights.h5')

classifier.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

# classifier = trainModel()

license_plate = Image.open('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Transfer Test/plate.jpg').resize((224,224))
print(license_plate)

license_plate = np.array(license_plate)/255.0
print(license_plate.shape)
print((license_plate[np.newaxis, ...]).shape)

result = classifier.predict(license_plate[np.newaxis, ...])
print(result)
print(result.shape)

classifier.summary()

image = convertBBoxImage(cutOutBBox(0), (224,224))
print(image.shape)
plt.show()