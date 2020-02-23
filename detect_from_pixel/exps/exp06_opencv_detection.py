import numpy as np
import cv2
import yappi
import datetime


model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#tb_writer = SummaryWriter(log_dir=ROOT_DIR + "/tensorboard/runs/segmentation_" + model_id)

colors = [((50, 0, 0), (255, 50, 50), "red"), ((0, 50, 0), (50, 255, 50), "green")]


image_orig = cv2.imread('images/layout_1.raw.png')
cv2.imshow("image_orig", image_orig)
npy = np.load("images/pcl_00.npy")
repeats = 1
yappi.start()
results = [None] * len(colors)
cnt = 0
images = []
for r in range(repeats):
    for idx, color in enumerate(colors):
        print("color={}".format(color[2]))
        result = image_orig.copy()
        image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        lower = np.array(color[0])
        upper = np.array(color[1])
        mask = cv2.inRange(image, lower, upper)
        mask = cv2.medianBlur(mask, 7)
        result = cv2.bitwise_and(result, result, mask=mask)
        #images.append((f"image_{idx}", image.transpose(2, 0, 1), cnt))
        #images.append((f"mask_{idx}", np.expand_dims(mask,0), cnt))
        #images.append((f"result_{idx}", result.transpose(2, 0, 1), cnt))
        cnt += 1
        cv2.imshow("mask {}".format(color[2]), mask)
        cv2.imshow("result {}".format(color[2]), result)

yappi.stop()
yappi.get_thread_stats().print_all()
t = yappi.get_clock_time()/repeats
print("t=" + str(t))

#for image in images:
#    tb_writer.add_image(*image)

#cv2.imshow('mask', mask)
#cv2.imshow('result', result)
cv2.waitKey()
#tb_writer.close()