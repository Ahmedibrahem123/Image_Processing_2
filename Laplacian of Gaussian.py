# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 23:32:43 2022

@author: ahmedibrahem
"""

#Laplacian of Gaussian (LoG)

import cv2
import matplotlib.pyplot as plt

# Read images 
img = cv2.imread('lena.jpg')
lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Grayscale the image 
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# First, the noise is reduced by Gaussian filtering 
gaussian = cv2.GaussianBlur(grayImage, (3,3), 0)

# Then the edge detection is done by Laplace operator 
dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize = 3)
LOG = cv2.convertScaleAbs(dst)

# Used to display Chinese labels normally 
plt.rcParams['font.sans-serif']=['SimHei']

# The graphics 
titles = [u' original image ', u'LOG operator ']
images = [lenna_img, LOG]
for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.show()