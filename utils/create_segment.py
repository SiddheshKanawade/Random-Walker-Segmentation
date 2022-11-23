from constants import COLORS
import numpy as np
import cv2 as cv
import logging
import random

from utils.helper import getVal
from utils.logger import logger

clicks = []


def get_seeds(noSegments, img, noPixels, labelledPixelsXY):
    def mouse_callback(event, x, y, flags, params):
        if event == 1:
            clicks.append([x, y])
            print(clicks)

    for n in range(noSegments):
        print("NOW WE ARE IN SEGMENT", n)
        cv.imshow("image", img)
        clicks = []
        cv.setMouseCallback('image', mouse_callback)
        while True:
            if len(clicks) == noPixels:
                break
            cv.waitKey(1)
        labelledPixelsXY.append(clicks)
        clicks = []
    logging.info("Completed making seeds")
    return labelledPixelsXY


def save_initial_markings(img, noSegments, labelledPixelsXY):
    imgCopy = np.array(img)
    for n in range(noSegments):
        for i in range(len(labelledPixelsXY[n])):
            cv.circle(imgCopy, (labelledPixelsXY[n][i][0],
                                labelledPixelsXY[n][i][1]), 2, COLORS[n+1], 3)
    return imgCopy


def get_transition_prob(img, cumilativeProbUpRightDownLeft):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            urdl = [getVal(y-1, x, img), getVal(y, x+1, img),
                    getVal(y+1, x, img), getVal(y, x-1, img)]
            nonNormalizedProbURDL = []
            for a in range(4):
                tt = np.mean(np.abs(urdl[a]-img[y, x, :]))
                tt = np.exp(-1*np.power(tt, 2))
                nonNormalizedProbURDL.append(tt)
            nonNormalizedProbURDL = np.array(nonNormalizedProbURDL)
            normalizedProbURDL = \
                nonNormalizedProbURDL / np.sum(nonNormalizedProbURDL)
            cumilativeProbUpRightDownLeft[y, x, 0] = normalizedProbURDL[0]
            for a in range(1, 4):
                cumilativeProbUpRightDownLeft[y, x, a] =\
                    cumilativeProbUpRightDownLeft[y, x, a-1] +\
                    normalizedProbURDL[a]
    return cumilativeProbUpRightDownLeft


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
