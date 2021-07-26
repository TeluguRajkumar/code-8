#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:59:37 2021

@author: rajkumar
"""

#check for certain RGB in image

##need to screen grab
#pip install Image
from PIL import Image
import Image, sys

x = 0
y = 0

im = Image.open("/Users/rajkumar/Desktop/view2.jpg")
pix = im.load()
print (im.size) #get width and height of the image for iterating over
while x < 1914:
    value = pix[x,y] #get RGBA value of the pixel of an image
    if value == (33, 179, 80):
        #call left_click(x,y)
        print (x,y)
    x = x + 1
    if x == 1914:
        x = 0
        y = y + 1
print ("Finished")
sys.exit()
##It finds the x and y coords of the RGB value in less than a second

#####Background removal of car image

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
from  PIL  import Image

img = cv.imread("/Users/rajkumar/Desktop/view2.jpg", cv.IMREAD_UNCHANGED)
original = img.copy()

l = int(max(5, 6))
u = int(min(6, 6))

ed = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.GaussianBlur(img, (21, 51), 3)
edges = cv.cvtColor(edges, cv.COLOR_BGR2GRAY)
edges = cv.Canny(edges, l, u)

_, thresh = cv.threshold(edges, 0, 255, cv.THRESH_BINARY  + cv.THRESH_OTSU)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=4)

data = mask.tolist()
sys.setrecursionlimit(10**8)
for i in  range(len(data)):
    for j in  range(len(data[i])):
        if data[i][j] !=  255:
            data[i][j] =  -1
        else:
            break
    for j in  range(len(data[i])-1, -1, -1):
        if data[i][j] !=  255:
            data[i][j] =  -1
        else:
            break
image = np.array(data)
image[image !=  -1] =  255
image[image ==  -1] =  0

mask = np.array(image, np.uint8)

result = cv.bitwise_and(original, original, mask=mask)
result[mask ==  0] =  255
cv.imwrite('bg.png', result)

img = Image.open('bg.png')
img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if item[0] ==  255  and item[1] ==  255  and item[2] ==  255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)

img.putdata(newData)
img.save("img.png", "PNG")



#### Car BAckgrounf replacement using OpenCV
import cv2
import numpy as np

# opencv loads the image in BGR, convert it to RGB
img = cv2.cvtColor(cv2.imread("/Users/rajkumar/Desktop/view1.jpg"),
                   cv2.COLOR_BGR2RGB)
lower_white = np.array([220, 220, 220], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
mask = cv2.inRange(img, lower_white, upper_white)  # could also use threshold
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
mask = cv2.bitwise_not(mask)  # invert mask

# load background (could be an image too)
bk = np.full(img.shape, 255, dtype=np.uint8)  # white bk

# get masked foreground
fg_masked = cv2.bitwise_and(img, img, mask=mask)

# get masked background, mask must be inverted 
mask = cv2.bitwise_not(mask)
bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

# combine masked foreground and masked background 
final = cv2.bitwise_or(fg_masked, bk_masked)
mask = cv2.bitwise_not(mask)  # revert mask to original
##If need black background set RHS to zero instead of 255
