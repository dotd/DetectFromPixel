"""
Exmaple taken from https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
"""
import cv2
import matplotlib as plt
from definitions_detect_from_pixel import ROOT_DIR


#python
bright = cv2.imread(ROOT_DIR + '/exmaples/rubic_cube/cube1.jpg')
dark = cv2.imread(ROOT_DIR + '/exmaples/rubic_cube/cube8.jpg')
plt



#python
brightLAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)