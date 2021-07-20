# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:06:08 2020

@author: Chapman Guan
"""

import numpy as np
import tensorflow as tf
import math


from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, GlobalMaxPool2D
from tensorflow.keras.models import Model, load_model


from keras.metrics import mean_squared_error

def model_keras(input_shape=(60, 60, 1), output_shape=6):
    
    X_input  = Input(input_shape)
    
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name="conv1")(X_input)
    X = Activation("relu")(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name="conv2")(X)
    X = Activation("relu")(X)
    
    X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name="conv3")(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name="conv4")(X)
    X = Activation("relu")(X)
    
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name="conv6")(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name="conv7")(X)
    X = Activation("relu")(X)
    
    X = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', name="conv8")(X)
    X = Activation("relu")(X)
    X = Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', name="conv9")(X)
    X = Activation("relu")(X)
    
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    
    X = Flatten()(X)
    
    X = Dense(8192, activation="relu", name="fc12")(X)
    X = Dense(2048, activation="relu", name="fc13")(X)
    
    X1 = Dense(1024, activation="relu", name="fc140")(X)
    X2 = Dense(1024, activation="relu", name="fc141")(X)
    X3 = Dense(1024, activation="relu", name="fc142")(X)
    X4 = Dense(1024, activation="relu", name="fc143")(X)
    X5 = Dense(1024, activation="relu", name="fc144")(X)
    X6 = Dense(1024, activation="relu", name="fc145")(X)
    
    X1 = Dense(256, activation="relu", name="fc150")(X1)
    X2 = Dense(256, activation="relu", name="fc151")(X2)
    X3 = Dense(256, activation="relu", name="fc152")(X3)
    X4 = Dense(256, activation="relu", name="fc153")(X4)
    X5 = Dense(256, activation="relu", name="fc154")(X5)
    X6 = Dense(256, activation="relu", name="fc155")(X6)
    
    X1 = Dense(1, activation="linear", name="fc160")(X1)
    X2 = Dense(1, activation="linear", name="fc161")(X2)
    X3 = Dense(1, activation="linear", name="fc162")(X3)
    X4 = Dense(1, activation="linear", name="fc163")(X4)
    X5 = Dense(1, activation="linear", name="fc164")(X5)
    X6 = Dense(1, activation="linear", name="fc165")(X6)
    
    model = Model(inputs=X_input, outputs=[X1,X2,X3,X4,X5,X6], name="model")
    
    return model

if __name__ == '__main__':
    model = model_keras(input_shape=(60, 60, 1), output_shape=6)
    model.summary()
    
    