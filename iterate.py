import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from FrameCrop import *

# df = pd.read_csv('goodCropFiles.csv')
# df = pd.read_csv('averageCropFiles.csv')
df = pd.read_csv('badCropFiles.csv')
files = df.iloc[:, 0]
frameCrop = FrameCrop(60, 20)

for i in range(0, len(files)):
    image = io.imread('data/' + files[i])

    # Show before and after cropping
    ax = plt.subplot(1, 2, 1)
    plt.tight_layout()
    ax.set_title('Original Image')
    plt.imshow(image, cmap='gray')

    ax = plt.subplot(1, 2, 2)
    plt.tight_layout()
    ax.set_title('FrameCrop')
    plt.imshow(frameCrop(image), cmap='gray')

    plt.show()

print('End of program')