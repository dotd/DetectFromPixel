from os import listdir
from os.path import isfile, join
from definitions_detect_from_pixel import ROOT_DIR
import matplotlib.pyplot as plt
import cv2
import h5py
import yappi
from detect_from_pixel.segmentation_utils import ColorDetector


# Load all images to memory
def load_images_to_memory(path_to_load, limit=int(1e6)):
    files = [f for f in listdir(path_to_load) if isfile(join(path_to_load, f))]
    files.sort()
    images = list()
    cnt = 0
    for i in range(0, (len(files)//4)*4, 4):
        if cnt >= limit:
            break
        fullpath_rgb = "{}/{}".format(path_to_load, files[i])
        fullpath_h5 = "{}/{}".format(path_to_load,files[i + 1])
        img = cv2.imread(fullpath_rgb)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fh5 = h5py.File(fullpath_h5, 'r')
        depth = fh5["depth"].value
        images.append((img, depth))
        if i % 100 == 0:
            print("load images {}/{}".format(i, len(files)))
        cnt += 1
        if cnt >= limit:
            break
    return images


#folder = "/data/20200219_121911/_D415_839112061357/" # Standing red box
#folder = "/data/20200220_122530/_D415_839112061357/" # Dark green small cube
#folder = "/data/20200220_122519/_D415_839112061357/" # Dark green small cube
folder = "/data/20200219_121407/_D415_839112060982/" # light green from above

path_to_load = "{}/{}/".format(ROOT_DIR, folder)
plt.figure(1)
cd = ColorDetector()
images = load_images_to_memory(path_to_load, 1000)
flag_show = True
yappi.start()
for (i, (img, depth)) in enumerate(images):

    blobs, blobs_depth = cd.detect(img, depth, method="normalize")

    if flag_show:
        print(i)
        plt.subplot(3, 2, 1)
        plt.imshow(img)
        plt.subplot(3, 2, 2)
        plt.imshow(depth)
        plt.subplot(3, 2, 3)
        plt.imshow(blobs[0])
        plt.subplot(3, 2, 4)
        plt.imshow(blobs_depth[0])
        if blobs[1] is not None:
            plt.subplot(3, 2, 5)
            plt.imshow(blobs[1])
            plt.subplot(3, 2, 6)
            plt.imshow(blobs_depth[1])
        else:
            print("Identification problem")

        plt.show(block=False)
        plt.pause(0.01)

yappi.stop()
yappi.get_thread_stats().print_all()
print("per repeat={}".format(yappi.get_clock_time()/len(images)))
print("total={}".format(str(yappi.get_clock_time())))
plt.show(block=True)