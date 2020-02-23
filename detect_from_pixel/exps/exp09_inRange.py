import numpy as np
import cv2

# 4 channels experiment
# Result: it is 4 channels

typ = np.uint8

img = np.array([[[0,0,0,0], [1,1,1,1]], [[2,2,2,2], [1,1,1,1]]], dtype=typ)*10
print(img.shape)
print(img.dtype)

low = np.array([1,1,1,1],dtype=typ)*5
high = np.array([1,1,1,1],dtype=typ)*15
print(low.dtype)
print(high.dtype)

mask = cv2.inRange(img, low, high)
print(mask)

