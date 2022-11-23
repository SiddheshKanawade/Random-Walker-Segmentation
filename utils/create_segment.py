from constants import COLORS
import numpy as np
import cv2 as cv
import logging
import random

from utils.helper import get_value
from utils.logger import logger

clicks = []


def get_seeds(total_segments, img, total_pixels, labels_algorithm):
    def mouse_callback(event, x, y, flags, params):
        if event == 1:
            clicks.append([x, y])
            print(clicks)

    for n in range(total_segments):
        print("NOW WE ARE IN SEGMENT", n)
        cv.imshow("image", img)
        clicks = []
        cv.setMouseCallback('image', mouse_callback)
        while True:
            if len(clicks) == total_pixels:
                break
            cv.waitKey(1)
        labels_algorithm.append(clicks)
        clicks = []
    logging.info("Completed making seeds")
    return labels_algorithm


def save_initial_markings(img, total_segments, labels_algorithm):
    imgCopy = np.array(img)
    for n in range(total_segments):
        for i in range(len(labels_algorithm[n])):
            cv.circle(imgCopy, (labels_algorithm[n][i][0],
                                labels_algorithm[n][i][1]), 2, COLORS[n+1], 3)
    return imgCopy


def get_transition_prob(img, cummulative_prob_neighbour):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            urdl = [get_value(y-1, x, img), get_value(y, x+1, img),
                    get_value(y+1, x, img), get_value(y, x-1, img)]
            non_normalized_proburdl = []
            for a in range(4):
                tt = np.mean(np.abs(urdl[a]-img[y, x, :]))
                tt = np.exp(-1*np.power(tt, 2))
                non_normalized_proburdl.append(tt)
            non_normalized_proburdl = np.array(non_normalized_proburdl)
            normalizedProbURDL = \
                non_normalized_proburdl / np.sum(non_normalized_proburdl)
            cummulative_prob_neighbour[y, x, 0] = normalizedProbURDL[0]
            for a in range(1, 4):
                cummulative_prob_neighbour[y, x, a] =\
                    cummulative_prob_neighbour[y, x, a-1] +\
                    normalizedProbURDL[a]
    return cummulative_prob_neighbour


def random_walker(segments, initiallyMarked, cumilativeProbUpRightDownLeft):
    for y in range(segments.shape[0]):
        for x in range(segments.shape[1]):
            if segments[y][x] == -1:
                yy = y
                xx = x

                while (initiallyMarked[yy, xx] == -1):
                    rv = random.random()
                    if cumilativeProbUpRightDownLeft[yy, xx, 0] > rv:
                        yy -= 1
                    elif cumilativeProbUpRightDownLeft[yy, xx, 1] > rv:
                        xx += 1
                    elif cumilativeProbUpRightDownLeft[yy, xx, 2] > rv:
                        yy += 1
                    else:
                        xx -= 1
                segments[y, x] = initiallyMarked[yy, xx]
        logger.info(f"Remaining Rows {y}")
    return segments
