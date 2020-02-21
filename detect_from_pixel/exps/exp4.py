import numpy as np
import torch
from scipy import misc
import cv2
import time
import yappi
import datetime

image = torch.tensor(misc.imread("layout_1.raw.png"), dtype=torch.int)
#image = cv2.imread("layout_1.raw.png")
boundaries = [([17, 15, 100], [50, 56, 200])]#, ([86, 31, 4], [220, 88, 50]), ([25, 146, 190], [62, 174, 250]), ([103, 86, 65], [145, 133, 128])]

images = []
for p in range(1):
    for (lower, upper) in boundaries:
        yappi.start()
        image2 = image.clone()
        lower = torch.tensor(lower, dtype=torch.uint8)
        upper = torch.tensor(upper, dtype=torch.uint8)
        idx = (image >= lower) & (image <= upper)
        idx2 = torch.prod(idx, axis=2)
        yappi.get_thread_stats().print_all()
        idx2 = torch.stack((idx2, idx2, idx2), 2)
        image2 = image2 * (1 - idx2)
        images.append(image2)

#
#

#yappi.get_func_stats().save('callgrind.out.' + datetime.datetime.now().isoformat("%Y%m%d_%H%M%S"), 'CALLGRIND')
