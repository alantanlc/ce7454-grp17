import matplotlib.pyplot as plt
import seaborn as sns
from dataset import CheXpertDataset
from skimage import io
import numpy as np
import pandas as pd

def isPoorlyCropped(sample, mean=60, std=20, lengthThreshold=50):
    top, bottom, left, right = 0, sample.shape[0] - 1, 0, sample.shape[1] - 1
    topMean, bottomMean, leftMean, rightMean = 0, 0, 0, 0
    topStd, bottomStd, leftStd, rightStd = 0, 0, 0, 0

    # Find top
    while topMean < mean or topStd < std:
        topMean = np.mean(sample[top, :])
        topStd = np.std(sample[top, :])
        top += 1
        if top >= lengthThreshold:
            return 1

    # Find bottom
    while bottomMean < mean or bottomStd < std:
        bottomMean = np.mean(sample[bottom, :])
        bottomStd = np.std(sample[bottom, :])
        bottom -= 1
        if sample.shape[0] - bottom >= lengthThreshold:
            return 1

    # Find left
    while leftMean < mean or leftStd < std:
        leftMean = np.mean(sample[:, left])
        leftStd = np.std(sample[:, std])
        left += 1
        if left >= lengthThreshold:
            return 1

    # Find right
    while rightMean < mean or rightStd < std:
        rightMean = np.mean(sample[:, right])
        rightStd = np.std(sample[:, right])
        right -= 1
        if sample.shape[1] - right >= lengthThreshold:
            return 1

    return 0

train = CheXpertDataset()
labels = train.labels_cols

# Distribution of well-cropped and poorly-cropped images (This takes about 30 mins to complete)
counts = [0, 0]
poorlyCroppedFiles = []
wellCroppedFiles = []
for i in range(len(train)):
    file = train.csv.iloc[i, 0]
    sample = io.imread('data/' + file)
    idx = isPoorlyCropped(sample)
    counts[idx] += 1

    if idx == 1:
        poorlyCroppedFiles.append(file)
    else:
        wellCroppedFiles.append(file)

# Write to csv
df = pd.DataFrame(poorlyCroppedFiles, columns=["Path"])
df.to_csv('poorlyCroppedFiles.csv', index=False)
df = pd.DataFrame(wellCroppedFiles, columns=["Path"])
df.to_csv('wellCroppedFiles.csv', index=False)

# Plot figure
plt.figure(figsize=(12, 6))
sns.barplot(['Well-cropped', 'Poorly-cropped'], counts, alpha = 0.9)
plt.xticks(rotation = 'vertical')
plt.xlabel('Crop', fontsize = 12)
plt.ylabel('Counts', fontsize = 12)
plt.savefig("well_poorly_cropped.png")

# Distribution of frontal and lateral images
frontal_lateral_counts = train.csv.iloc[:, 3].value_counts()
plt.figure(figsize = (12,6))
sns.barplot(frontal_lateral_counts.index, frontal_lateral_counts.values, alpha = 0.9)
plt.xticks(rotation = 'vertical')
plt.xlabel('Fracture labels', fontsize = 12)
plt.ylabel('Counts', fontsize = 12)

# Distribution of target variable
label_counts = []
plt.figure(figsize = (12,6))
for label in labels:
    count = train.csv.loc[:, label].value_counts().values[0]
    label_counts.append(count)
sns.barplot(labels, label_counts, alpha = 0.9)
plt.xticks(rotation = 'vertical')
plt.xlabel('Classes', fontsize=10)
plt.ylabel('Counts', fontsize=10)

# Distribution of the variable 'age'
ax = plt.figure(figsize=(30, 8))
sns.countplot(train.csv.Age)
axis_font = {'fontname': 'Arial', 'size':'24'}
plt.xlabel('age', **axis_font)
plt.ylabel('Count', **axis_font)

plt.show()
print('End of program')