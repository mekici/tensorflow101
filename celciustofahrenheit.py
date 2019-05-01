#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:28:59 2019

@author: multiproxy
"""

import tensorflow as tf
import numpy as np

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

