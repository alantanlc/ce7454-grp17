import matplotlib.pyplot as plt
import seaborn as sns
from dataset import CheXpertDataset
from skimage import io
import numpy as np
import pandas as pd

def getCropQuality(sample, mean=60, std=20, averageThreshold=30, badThreshold=60):
    top, bottom, left, right = 0, sample.shape[0] - 1, 0, sample.shape[1] - 1
    topMean, bottomMean, leftMean, rightMean = 0, 0, 0, 0
    topStd, bottomStd, leftStd, rightStd = 0, 0, 0, 0

    # Find top
    while topMean < mean or topStd < std and topMean < 100:
        topMean = np.mean(sample[top, :])
        topStd = np.std(sample[top, :])
        top += 1
    if top >= badThreshold:
        return 2
    elif top >= averageThreshold:
        return 1

    # Find bottom
    while bottomMean < mean or bottomStd < std and bottomMean < 100:
        bottomMean = np.mean(sample[bottom, :])
        bottomStd = np.std(sample[bottom, :])
        bottom -= 1
    if sample.shape[0] - bottom >= badThreshold:
        return 2
    elif sample.shape[0] - bottom >= averageThreshold:
        return 1

    # Find left
    while leftMean < mean or leftStd < std and leftMean < 100:
        leftMean = np.mean(sample[:, left])
        leftStd = np.std(sample[:, left])
        left += 1
    if left >= badThreshold:
        return 2
    elif left >= averageThreshold:
        return 1

    # Find right
    while rightMean < mean or rightStd < std and rightMean < 100:
        rightMean = np.mean(sample[:, right])
        rightStd = np.std(sample[:, right])
        right -= 1
    if sample.shape[1] - right >= badThreshold:
        return 2
    elif sample.shape[1] - right >= averageThreshold:
        return 1

    return 0

train = CheXpertDataset()
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

# Number of images under each label
colSums = train.csv.loc[:, labels].sum().values
ax = sns.barplot(labels, colSums)
plt.title("Images in each category",)
plt.xlabel('Pathology')
plt.ylabel('Number of images')
# adding the text labels
rects = ax.patches
for rect, label in zip(rects, colSums):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, label, ha='center', va='bottom')
plt.show()

# Distribution of frontal and lateral images
frontal_lateral_counts = train.csv.iloc[:, 3].value_counts()
plt.figure()
ax = sns.barplot(frontal_lateral_counts.index, frontal_lateral_counts.values)
plt.title('Images in each view')
plt.xlabel('Image View')
plt.ylabel('Number of images')
# adding the text labels
rects = ax.patches
label_values = frontal_lateral_counts.values
for rect, label in zip(rects, label_values):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, label, ha='center', va='bottom')
plt.show()

# Number of comments having multiple labels
rowSums = train.csv.loc[:, labels].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
plt.figure()
ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
plt.title("Images having multiple labels")
plt.xlabel("Number of labels")
plt.ylabel("Number of images")
# adding the text labels
rects = ax.patches
label_values = multiLabel_counts.values
for rect, label in zip(rects, label_values):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height+5, label, ha='center', va='bottom')
plt.show()

# Distribution of the variable 'age'
plt.figure()
ax = sns.countplot(train.csv.Age)
plt.title('Images in each age')
plt.xlabel('Age')
plt.ylabel('Number of images')
plt.show()

# Distribution of well-cropped and poorly-cropped images (This takes about 30 mins to complete)
# counts = [0, 0, 0]
# goodCropFiles = []
# averageCropFiles = []
# badCropFiles = []
# for i in range(len(train)):
#     file = train.csv.iloc[i, 0]
#     sample = io.imread('data/' + file)
#     idx = getCropQuality(sample)
#     counts[idx] += 1
#
#     if idx == 0:
#         goodCropFiles.append(file)
#     elif idx == 1:
#         averageCropFiles.append(file)
#     else:
#         badCropFiles.append(file)
# # Write to csv
# df = pd.DataFrame(goodCropFiles, columns=["Path"])
# df.to_csv('goodCropFiles.csv', index=False)
# df = pd.DataFrame(averageCropFiles, columns=["Path"])
# df.to_csv('averageCropFiles.csv', index=False)
# df = pd.DataFrame(badCropFiles, columns=["Path"])
# df.to_csv('badCropFiles.csv', index=False)
# # Plot figure
# plt.figure(figsize=(12, 6))
# sns.barplot(['Good', 'Average', 'Bad'], counts, alpha = 0.9)
# plt.xticks(rotation = 'vertical')
# plt.xlabel('Crop Quality')
# plt.ylabel('Counts')
# plt.savefig("crop_quality.png")

print('End of program')