from utils.logger import logger
import numpy as np
import cv2 as cv
from constants import COLORS, FACTOR


def down(x):
    return int(x*FACTOR)


def up(x):
    return int(x/FACTOR)


def get_value(y, x, ar):
    if x < 0 or y < 0 or y >= ar.shape[0] or x >= ar.shape[1]:
        return np.array([-1000.0, -1000.0, -1000.0])
    else:
        return ar[y, x, :]


def get_output_image(original_image, segments):
    outputImg = np.array(original_image)
    for y in range(outputImg.shape[0]):
        for x in range(outputImg.shape[1]):
            outputImg[y, x] = COLORS[segments[down(y)-1, down(x)-1]]
    return outputImg


def generate_images(args, image_copy, output_image):
    logger.info(f"Storing Images at {args.imagePath[:4]}")
    cv.imwrite(f"initial.jpg", image_copy)
    cv.imwrite("segment.jpg", output_image)
    cv.imwrite("final.jpg",
               np.concatenate((image_copy, output_image), axis=1))
