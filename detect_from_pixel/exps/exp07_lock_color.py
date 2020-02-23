import numpy as np
import cv2
import yappi
import datetime
import detect_from_pixel.segmentation_utils as su


model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

colors = list()
colors.append(([np.array([50, 0, 0]), np.array([255, 50, 50]), "red"]))
colors.append(([np.array([0, 50, 0]), np.array([50, 255, 50]), "green"]))

image_orig = cv2.imread('images/layout_1.raw.png')
cv2.imshow("image_orig", image_orig)

repeats = 10
images = []
yappi.start()
for repeat in range(repeats):
    for idx, color in enumerate(colors):
        print("repeat={}".format(repeat))
        #print(f"color={color[2]}")
        result = image_orig.copy()
        image_cv2 = image_orig.copy()
        image_rgb = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        lower = color[0]
        upper = color[1]
        print("min={lower} max={upper}".format(lower, upper))
        mask = cv2.inRange(image_rgb, lower, upper)
        mask = cv2.medianBlur(mask, 7)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        biggest_contour, biggest_score, biggest_values = su.identify_biggest_contour_bb(contours, convex_hull_flag=True)
        values = su.get_values_in_contour(image_cv2, biggest_contour)
        color[0] = np.amin(values, axis=0)
        color[1] = np.amax(values, axis=0)



        # Get the biggest bounding box
        # biggest_bb = image_cv2[biggest_values[1]:biggest_values[3], biggest_values[0]:biggest_values[2], :]




        #biggest_contour_mask_convex_hull = mask_from_contours(image_cv2, [biggest_convex_hull])
        #result = cv2.bitwise_and(result, result, mask=biggest_contour_mask_convex_hull)
        #result = result[biggest_values[1]:biggest_values[3], biggest_values[0]:biggest_values[2], :]
        #for channel in range(3):
        #    lower[channel] = np.min(result[result[:, :, channel] > 0])
        #    upper[channel] = np.max(result[result[:, :, channel] > 0])

        # Correcting

        #images.append([result, f"hull_{color[2]}"])

        #biggest_contour_mask = mask[biggest_values[1]:biggest_values[3], biggest_values[0]:biggest_values[2]]
        #biggest_contour = cv2.bitwise_and(biggest_contour, biggest_contour, mask=biggest_contour_mask)
        #images.append([biggest_contour, f"{color[2]}"])


        #image_with_countours = cv2.drawContours(image_orig.copy(), contours, -1, (0, 255, 0), 1)

        #cv2.imshow(f"mask {color[2]}", mask)
        #cv2.imshow(f"result {color[2]}", result)
        #cv2.imshow(f"image_orig {color[2]}", image_with_countours)

yappi.stop()
yappi.get_thread_stats().print_all()
print("per repeat={}".format(str(yappi.get_clock_time()/repeats)))
print("total={}".format(str(yappi.get_clock_time())))
for image_rgb in images:
    cv2.imshow(image_rgb[1], image_rgb[0])

cv2.waitKey()
