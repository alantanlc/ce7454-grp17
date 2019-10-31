import numpy as np
from PIL import Image

class FrameCrop:
    """ Perform frame crop to remove black borders """

    def __init__(self, threshold=0):
        assert isinstance(threshold, (int, float)) and threshold >= 0
        self.threshold = threshold

    def __call__(self, sample):
        sample = np.asarray(sample)

        top, bottom, left, right = 0, sample.shape[0]-1, 0, sample.shape[1]-1
        topThres, bottomThres, leftThres, rightThres = 0, 0, 0, 0

        # Find top
        while topThres < self.threshold:
            topThres = np.mean(sample[top, :])
            top += 1

        # Find bottom
        while bottomThres < self.threshold:
            bottomThres = np.mean(sample[bottom, :])
            bottom -= 1

        # Find left
        while leftThres < self.threshold:
            leftThres = np.mean(sample[:, left])
            left += 1

        # Find right
        while rightThres < self.threshold:
            rightThres = np.mean(sample[:, right])
            right -= 1

        return Image.fromarray(sample[top:bottom, left:right])