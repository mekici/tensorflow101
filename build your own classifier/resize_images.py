#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 00:04:03 2019

@author: multiproxy
"""

from PIL import Image
import os
def rescale_images(directory, width,height):
    for img in os.listdir(directory):
        im = Image.open(directory+img)
        im=im.convert('RGB')
        im_resized = im.resize((width,height), Image.ANTIALIAS)
        im_resized.save(directory+img)

directory="elma/"
width=32
height=32
rescale_images(directory, width, height)