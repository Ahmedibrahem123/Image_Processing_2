# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:40:11 2022

@author: ahmedibrahem
"""

#Sobel

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images 
img = cv2.imread('lena.jpg')
lenna_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Grayscale the image 
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Sobel operator 
x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)  
y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1) 
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# Used to display Chinese labels normally 
plt.rcParams['font.sans-serif']=['SimHei']

# The graphics 
titles = [u' original image ', u'Sobel operator ']
images = [lenna_img, Sobel]
for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.show()