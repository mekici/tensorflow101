#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:20:00 2019

@author: multiproxy
"""
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
num_channel=1

loaded_model=load_model('model.hdf5')

test_image = cv2.imread('rider-13.jpg')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
K.set_image_dim_ordering('th')

if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
        
        
names = ['kediler','köpekler','atlar','insanlar']
result=np.argmax(loaded_model.predict(test_image))


print("bu fotoğraf şu sınıfa ait:", names[result])
