from loadData import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.models import save_model
from sklearn.model_selection import train_test_split
from keras.utils import normalize

#Notes:
# Kan eventueel model als een file exporteren.

def trainModel():
    X, y = loadDataSet()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    X_train_scaled = normalize(X_train, axis=1)
    X_test_scaled = normalize(X_test, axis=1)
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)
    print(X[0].shape)
    print(X_train_scaled)

    feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

    pretrained_model_without_top_layer = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

    model = tf.keras.Sequential([
    pretrained_model_without_top_layer,
    # Dense layer is een neuron, want output is plate of geen plate.
    # Sigmoid en geen softmax, want sigmoid is voor binaire classes en softmax is voor meerde classes.
    tf.keras.layers.Dense(1, activation='sigmoid') # by default, activation=None, so "logits" output.
    ])                # Dense implements the operation: output = activation(dot(input, kernel) + bias)

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    
    history = model.fit(X_train_scaled, y_train, epochs=20, validation_data=(X_test_scaled, y_test))  # commented out, because we’re gonna train with augmentation instead, below:

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    model.evaluate(X_test_scaled,y_test)

    model_json = model.to_json()
    with open('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_arch.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('C:/Users/tjezv/OneDrive/Desktop/Vision Opdrachten/Eindopdracht/my_model_weights.h5')

    return model