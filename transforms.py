from __future__ import print_function, division

from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import CheXpertDataset
from HistogramEqualize import *
from MedianBlur import *

# Load datasets
image_datasets = {x: CheXpertDataset(training=(x == 'train')) for x in ['train', 'val']}

# Individual transforms
mean, std = 127.8989, 74.69
resize = transforms.Resize(365)
randomCrop = transforms.RandomCrop(320)
centerCrop = transforms.CenterCrop(320)
medianBlur = MedianBlur(3)
histEq = HistogramEqualize()
toTensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[mean], std=[std])
toPILImage = transforms.ToPILImage()

# Apply each of the train transform on sample
fig = plt.figure()
sample = image_datasets['train'][0]
titles = ['Resize', 'Random Crop', 'Median Blur', 'Histogram Equalization', 'Normalize', 'Compose']

for i, tsfrm in enumerate([transforms.Compose([toPILImage, resize]),
                           transforms.Compose([toPILImage, randomCrop]),
                           transforms.Compose([toPILImage, medianBlur]),
                           transforms.Compose([toPILImage, histEq]),
                           transforms.Compose([normalize, toPILImage]),
                           transforms.Compose([toPILImage, resize, randomCrop, medianBlur, histEq, toTensor, normalize, toPILImage])]):
    image = sample[0]
    transformed_sample = tsfrm(image)

    ax = plt.subplot(2, 3, i+1)
    plt.tight_layout()
    ax.set_title(titles[i])
    plt.imshow(transformed_sample, cmap='gray')

# Apply each of the valid transform on sample
fig = plt.figure()
sample = image_datasets['train'][1]
titles = ['Resize', 'Center Crop', 'Median Blur', 'Histogram Equalization', 'Normalize', 'Compose']
for i, tsfrm in enumerate([transforms.Compose([toPILImage, resize]),
                           transforms.Compose([toPILImage, centerCrop]),
                           transforms.Compose([toPILImage, medianBlur]),
                           transforms.Compose([toPILImage, histEq]),
                           transforms.Compose([normalize, toPILImage]),
                           transforms.Compose([toPILImage, resize, centerCrop, medianBlur, histEq, toTensor, normalize, toPILImage])]):
    image = sample[0]
    transformed_sample = tsfrm(image)

    ax = plt.subplot(2, 3, i+1)
    plt.tight_layout()
    ax.set_title(titles[i])
    plt.imshow(transformed_sample, cmap='gray')
plt.show()

print('End of program')