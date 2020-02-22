import numpy as np
import cv2


class ColorDetector:

    colors_rgb = [(np.array([50, 0, 0]), np.array([255, 50, 50]), "red"),
                  (np.array([0, 120, 0]), np.array([70, 255, 70]), "green")]

    colors_normalize = [(np.array([0.3, 0, 0, 0.2]), np.array([1, 0.2, 0.2, 0.8]), "red"),
                        (np.array([0, 0.3, 0, 0.2]), np.array([0.4, 1, 0.3, 0.9]), "green")]

    def __init__(self):
        pass


    def get_rgb2_threshold(self, threshold):
        color_rgb2 = list()
        # Green
        rgb = [16, 158, 40]
        minRGB = np.array([rgb[0] - threshold, rgb[1] - threshold, rgb[2] - threshold])
        maxRGB = np.array([rgb[0] + threshold, rgb[1] + threshold, rgb[2] + threshold])
        color_rgb2.append((minRGB, maxRGB, "green"))

        return color_rgb2

    @staticmethod
    def rgb_detect(img, color_low, color_high):
        mask = cv2.inRange(img, color_low, color_high)
        mask = cv2.medianBlur(mask, 7)
        return mask

    @staticmethod
    def normalize_detect(img_orig, color_low, color_high):
        img = img_orig.copy().astype(np.float16)
        sum = np.sum(img, 2)
        res = np.zeros(shape=(img.shape[0], img.shape[1], 4))
        for i in range(3):
            res[:, :, i] = img[:, :, i] / sum
        res[:, :, 3] = sum / (3 * 256)
        mask = cv2.inRange(res, color_low, color_high)
        mask = cv2.medianBlur(mask, 7)
        return mask

    @staticmethod
    def get_colors_by_method(method):
        if method == "rgb":
            colors = ColorDetector.colors_rgb
        elif method == "normalize":
            colors = ColorDetector.colors_normalize
        return colors

    def detect(self, img_orig, img_depth, method):
        img_colors = list()
        depth = list()
        colors = self.get_colors_by_method(method)
        for idx, color in enumerate(colors):
            img = img_orig
            if method == "rgb":
                mask = self.rgb_detect(img, color[0], color[1])
            elif method == "normalize":
                mask = self.normalize_detect(img, color[0], color[1])
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            biggest_contour, biggest_score, biggest_values = identify_biggest_contour_bb(contours,
                                                                                            convex_hull_flag=True)
            biggest_bb = None
            biggest_depth = None
            if biggest_contour is not None:
                biggest_bb = img_orig[biggest_values[1]:biggest_values[3],
                             biggest_values[0]:biggest_values[2], :]
                if depth is not None:
                    biggest_depth = img_depth[biggest_values[1]:biggest_values[3],
                                    biggest_values[0]:biggest_values[2]]

            img_colors.append(biggest_bb)
            depth.append(biggest_depth)
        return img_colors, depth


def get_values_in_contour(image, contour):
    contour_mask = get_bw_mask_from_contours(image, [contour])
    N = contour_mask.shape[0] * contour_mask.shape[1]
    image = image.reshape(N, -1)
    contour_mask = contour_mask.reshape(-1)
    values = image[contour_mask == 255, :]
    return values


def get_mask_from_contour(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    return mask


def mask_from_contours_boolean(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.bool)
    mask = cv2.drawContours(mask, contours, -1, True, -1)
    return mask


def get_bw_mask_from_contours(ref_img, contours):
    mask = get_mask_from_contour(ref_img, contours)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


def identify_biggest_contour_bb(contours, convex_hull_flag=False):
    # Identify the biggest contour by the bounding box
    biggest_score = -1
    biggest_bb = None
    biggest_contour = None
    for idx_contour, contour in enumerate(contours):
        # getting bounding box
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > biggest_score:
            biggest_bb = (x, y, x + w, y + h)
            biggest_score = w * h
            biggest_contour = contour
    if convex_hull_flag and biggest_contour is not None:
        biggest_contour = cv2.convexHull(biggest_contour, False)
    return biggest_contour, biggest_score, biggest_bb


def process_contour(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, w, h), w * h, contour, cv2.convexHull(contour, False)


def identify_sorted_contours_bb(contours):
    contours_rect = [process_contour(contour) for contour in contours]
    contours_rect = sorted(contours_rect, key=lambda a: a[1])
    return contours_rect


