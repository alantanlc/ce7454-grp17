import numpy as np
from PIL import Image

class FrameCrop:
    """ Perform frame crop to remove black borders """

    def __init__(self, mean=60, std=20):
        assert isinstance(mean, (int, float)) and mean >= 0
        assert  isinstance(std, (int, float)) and std >= 0
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample = np.asarray(sample)

        top, bottom, left, right = 0, sample.shape[0]-1, 0, sample.shape[1]-1
        topMean, bottomMean, leftMean, rightMean = 0, 0, 0, 0
        topStd, bottomStd, leftStd, rightStd = 0, 0, 0, 0

        # Find top
        while topMean < self.mean or topStd < self.std:
            topMean = np.mean(sample[top, :])
            topStd = np.std(sample[top, :])
            top += 1

        # Find bottom
        while bottomMean < self.mean or bottomStd < self.std:
            bottomMean = np.mean(sample[bottom, :])
            bottomStd = np.std(sample[bottom, :])
            bottom -= 1

        # Find left
        while leftMean < self.mean or leftStd < self.std:
            leftMean = np.mean(sample[:, left])
            leftStd = np.std(sample[:, left])
            left += 1

        # Find right
        while rightMean < self.mean or rightStd < self.std:
            rightMean = np.mean(sample[:, right])
            rightStd = np.std(sample[:, right])
            right -= 1

        # print(topStd, bottomStd, leftStd, rightStd)
        # print(top, sample.shape[0]-bottom, left, sample.shape[1]-right)

        return Image.fromarray(sample[top:bottom, left:right])