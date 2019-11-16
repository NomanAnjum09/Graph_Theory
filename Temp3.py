#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:53:50 2019

@author: root
"""
#!/usr/bin/python
from os import listdir
from PIL import Image as PImage

def loadImages(path):
    # return array of images

    img = PImage.open(path)
    return img

path = "/root/Pictures/kruskal.png"

# your images in an array
imgs = loadImages(path)

imgs.show()