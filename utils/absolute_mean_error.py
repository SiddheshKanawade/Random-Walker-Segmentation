import numpy as np


def absolute_mean_error(output_image, output_image_skimage):
    output_image = np.array(output_image, dtype=np.float32)
    output_image_skimage = np.array(output_image_skimage, dtype=np.float32)
    err = np.sum(abs(output_image-output_image_skimage))
    err /= (float(output_image.shape[0] * output_image.shape[1])*3)
    return err
