# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:43:42 2022

@author: ahmedibrahem
"""

#Roberts 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read images 
img = cv2.imread('lena.jpg')
lenna_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Grayscale the image 
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Roberts operator 
kernelx = np.array([[-1,0],[0,1]], dtype=int)
kernely = np.array([[0,-1],[1,0]], dtype=int)
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

# turn uint8 
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX,0.5,absY,0.5,0)

# Used to display Chinese labels normally 
plt.rcParams['font.sans-serif']=['SimHei']

# The graphics 
titles = [u' original image ', u'Roberts operator ']
images = [lenna_img, Roberts]
for i in range(2):
    plt.subplot((1,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.show()