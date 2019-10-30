import cv2
import numpy as np
import PIL

class HistogramEqualize(object):
    """ Perform histogram equalization to improve contrast of the image."""

    def __call__(self, sample):
        return PIL.Image.fromarray(cv2.equalizeHist(np.asarray(sample)))