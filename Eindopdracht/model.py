from loadData import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.models import save_model

def trainModel():
    X, y = loadDataSet()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255

    print(X[0].shape)

    feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

    pretrained_model_without_top_layer = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    model = tf.keras.Sequential([
    pretrained_model_without_top_layer,
    # Dense layer is een neuron, want output is plate of geen plate.
    tf.keras.layers.Dense(1) # by default, activation=None, so "logits" output.
    ])                # Dense implements the operation: output = activation(dot(input, kernel) + bias) 

    model.summary()

    model.build() # initialises the weights. Not needed when using model.fit.

    model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

    model.fit(X_train_scaled, y_train, epochs=2)  # commented out, because we’re gonna train with augmentation instead, below:

    model.evaluate(X_test_scaled,y_test)

    model_json = model.to_json()
    with open('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_arch.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_weights.h5')