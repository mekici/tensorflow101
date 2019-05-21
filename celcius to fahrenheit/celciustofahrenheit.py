#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:28:59 2019

@author: multiproxy
"""

import tensorflow as tf
import numpy as np
import os 


celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38],
                      dtype=float)#input
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100],
                      dtype=float)#output

for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenhet".format(c, fahrenheit_a[i]))
    
ly = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([ly])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
train = model.fit(celsius_q, fahrenheit_a, epochs=1000, verbose=False)
print("Finished training the model")

print(model.predict([100.0]))

print("These are the layer variables: {}".format(ly.get_weights()))

cwd = os.getcwd()

path = "/models/f2c"

path = cwd.join(path) 


tf.contrib.saved_model.save_keras_model(model, cwd)

model.save('my_model.h5')
model.save_weights('my_model_weights.h5')


"""saved_model_dir = "/tmp/models/f2c/1557951646"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

tflite_model=converter.convert()

open("/tmp/model.tflite","wb").write(tflite_model)"""


