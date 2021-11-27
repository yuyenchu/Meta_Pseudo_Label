import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, models


def get_model(input_shape=(28,28,3), classes=10):
    initializer = keras.initializers.RandomUniform(minval=0., maxval=1.)
    # return tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=True, weights=None, classes=classes, alpha=0.01)
    
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer, input_shape=input_shape))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(32, activation='relu', kernel_initializer=initializer))
    # model.add(layers.Dense(64, activation='relu', kernel_initializer=initializer))
    # model.add(layers.Dense(10, activation='softmax', kernel_initializer=initializer))
    # return model

    model = tf.keras.models.Sequential([
        layers.Conv2D(8, (3,3), activation='relu',input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(classes, activation='softmax')
    ])
    return model