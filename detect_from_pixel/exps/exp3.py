from scipy import misc
import torch
import time

import numpy as np
print("Load image")
image = misc.imread("layout_1.raw.png")
print("End load")
print(image.max())
print(image.min())
print("Finish min,max")

# Doing k-means
image_t = torch.tensor(image, dtype=torch.int)
centers = torch.tensor([[[255, 0, 0]], [[0, 255, 0]]])

r = torch.abs(image_t - centers[0])
g = torch.abs(image_t - centers[1])


print("")