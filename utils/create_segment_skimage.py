from constants import COLORS, FACTOR
from skimage import io, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import exposure
from skimage.segmentation import random_walker as random_walker_inbuilt
import cv2 as cv
from utils.helper import down


def random_walker_skimage(args, noSegments, labelledPixelsXY, imgOriginal, imgCopy, outputImg, segments):
    img = cv.imread(args.imagePath)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img/255.0
    img = cv.resize(img, (int(img.shape[1]*FACTOR)+1,
                          int(img.shape[0]*FACTOR)+1))
    markers = np.zeros(img.shape, dtype=np.uint)
    for s in range(noSegments):
        for a in range(len(labelledPixelsXY[s])):
            x_coord = int(labelledPixelsXY[s][a][0]*FACTOR+1)
            y_coord = int(labelledPixelsXY[s][a][1]*FACTOR+1)
            markers[y_coord, x_coord] = s+1
    labels = random_walker_inbuilt(img, markers, beta=250000, mode='bf')
    built_in_outputImg = np.array(imgOriginal)
    for y in range(built_in_outputImg.shape[0]):
        for x in range(built_in_outputImg.shape[1]):
            built_in_outputImg[y, x] = COLORS[labels[down(y), down(x)]]
    cv.imwrite("comparison.jpg", np.concatenate(
        (imgCopy, outputImg, built_in_outputImg), axis=1))
    return built_in_outputImg
