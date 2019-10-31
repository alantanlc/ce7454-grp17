from PIL.ImageOps import equalize

class HistogramEqualize:
    """ Perform histogram equalization to improve contrast of the image."""

    def __call__(self, sample):
        return equalize(sample)