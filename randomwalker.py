from asyncio.log import logger
from constants import COLORS, FACTOR
import cv2 as cv
import argparse
import numpy as np
import random
from utils.absolute_mean_error import absolute_mean_error

from utils.create_segment import get_seeds, get_transition_prob, random_walker, save_initial_markings
from utils.create_segment_skimage import random_walker_skimage
from utils.helper import *

# Add try and expect error method in them
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--imagePath", dest="imagePath", type=str)
    args.add_argument("-n", "--totalSegments", dest="totalSegments", type=int)
    args.add_argument("-p", "--totalPixels", dest="totalPixels", type=int)
    args = args.parse_args()

    img = cv.imread(args.imagePath)
    noSegments = args.totalSegments
    noPixels = args.totalPixels

    labelledPixelsXY = []

    labelledPixelsXY = get_seeds(noSegments, img, noPixels, labelledPixelsXY)
    imgCopy = save_initial_markings(img, noSegments, labelledPixelsXY)

    # >>>>>>> Resize the image to save computational time
    imgOriginal = np.array(img)
    img = img/255.0
    img = cv.resize(img, (int(img.shape[1]*FACTOR)+1,
                          int(img.shape[0]*FACTOR)+1))

    initiallyMarked = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
    initiallyMarked.fill(-1)
    segments = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
    segments.fill(-1)
    cumilativeProbUpRightDownLeft = np.zeros((img.shape[0],
                                              img.shape[1], 4), dtype=np.float)

    # Generate the transition probabilites based on pixel similarity
    cumilativeProbUpRightDownLeft = get_transition_prob(
        img, cumilativeProbUpRightDownLeft)

    for s in range(noSegments):
        for a in range(len(labelledPixelsXY[s])):
            initiallyMarked[down(labelledPixelsXY[s][a][1]),
                            down(labelledPixelsXY[s][a][0])] = s+1
            segments[down(labelledPixelsXY[s][a][1]),
                     down(labelledPixelsXY[s][a][0])] = s+1

    # Random Walker Algorithm
    segments_cp = segments
    assert segments_cp.all() == segments.all()
    segments_rw = random_walker(segments, initiallyMarked,
                                cumilativeProbUpRightDownLeft)

    outputImg = get_output_image(imgOriginal, segments_rw)
    outputImg_skimage = random_walker_skimage(
        args, noSegments, labelledPixelsXY, imgOriginal, imgCopy, outputImg, segments_cp)

    generate_images(args, imgCopy, outputImg)

    logger.info(f"Completed the segmentation")
    logger.info("")
    logger.info("Calculating Error")
    error = absolute_mean_error(outputImg, outputImg_skimage)

    logger.info(f"The mean absolute error is {error}")
