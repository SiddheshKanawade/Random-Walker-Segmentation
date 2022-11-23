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

try:

    if __name__ == "__main__":
        args = argparse.ArgumentParser()
        args.add_argument("-i", "--imagePath", dest="imagePath", type=str)
        args.add_argument("-n", "--totalSegments",
                          dest="totalSegments", type=int)
        args.add_argument("-p", "--totalPixels", dest="totalPixels", type=int)
        args = args.parse_args()

        img = cv.imread(args.imagePath)
        total_segments = args.totalSegments
        total_pixels = args.totalPixels

        labels_algorithm = []

        labels_algorithm = get_seeds(
            total_segments, img, total_pixels, labels_algorithm)
        imgCopy = save_initial_markings(img, total_segments, labels_algorithm)

        # Resize the image to save computational time
        original_image = np.array(img)
        img = img/255.0
        img = cv.resize(img, (int(img.shape[1]*FACTOR)+1,
                              int(img.shape[0]*FACTOR)+1))

        initial_markings = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
        initial_markings.fill(-1)
        segments = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
        segments.fill(-1)
        cummulative_prob_neighbour = np.zeros((img.shape[0],
                                               img.shape[1], 4), dtype=np.float)

        # Generate the transition probabilites based on pixel similarity
        cummulative_prob_neighbour = get_transition_prob(
            img, cummulative_prob_neighbour)

        for s in range(total_segments):
            for a in range(len(labels_algorithm[s])):
                initial_markings[down(labels_algorithm[s][a][1]),
                                 down(labels_algorithm[s][a][0])] = s+1
                segments[down(labels_algorithm[s][a][1]),
                         down(labels_algorithm[s][a][0])] = s+1

        # Random Walker Algorithm
        segments_rw = random_walker(segments, initial_markings,
                                    cummulative_prob_neighbour)

        output_image = get_output_image(original_image, segments_rw)
        output_image_skimage = random_walker_skimage(
            args, total_segments, labels_algorithm, original_image, imgCopy, output_image)

        generate_images(args, imgCopy, output_image)

        logger.info(f"Completed the segmentation")
        logger.info("")
        logger.info(f"Total Segments used: {total_segments}")
        logger.info(f"Total Pixels used: {total_pixels}")
        image_name = args.imagePath.split("/")[-1]
        logger.info(f"Image Used: {image_name}")
        logger.info("Calculating Error")
        error = absolute_mean_error(output_image, output_image_skimage)

        logger.info(f"The mean absolute error is {error}")
except Exception as e:
    logger.exception(e)
