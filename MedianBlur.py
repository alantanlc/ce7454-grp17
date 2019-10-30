import cv2
import numpy as np
import PIL

class MedianBlur:
    """ Perform median filtering to remove salt and pepper noise from image. """
    def __init__(self, ksize):
        assert isinstance(ksize, int) and ksize > 1 and ksize % 2 == 1
        self.ksize = ksize

    def __call__(self, sample):
        return PIL.Image.fromarray(cv2.medianBlur(np.asarray(sample), self.ksize))