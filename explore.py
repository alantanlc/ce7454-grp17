import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
'exec(%matplotlib inline)'

train = pd.read_csv('data/CheXpert-v1.0-small/train.csv')
test = pd.read_csv('data/CheXpert-v1.0-small/valid.csv')
labels = train.columns.values[-14:]
print(labels)

# Distribution of frontal and lateral images
frontal_lateral_counts = train[train.columns.values[3]].value_counts()
plt.figure(figsize = (12,6))
sns.barplot(frontal_lateral_counts.index, frontal_lateral_counts.values, alpha = 0.9)
plt.xticks(rotation = 'vertical')
plt.xlabel('Fracture labels', fontsize = 12)
plt.ylabel('Counts', fontsize = 12)

# Distribution of target variable
label_counts = []
plt.figure(figsize = (12,6))
for label in labels:
    count = train[label].value_counts().values[0]
    label_counts.append(count)
sns.barplot(labels, label_counts, alpha = 0.9)
plt.xticks(rotation = 'vertical')
plt.xlabel('Classes', fontsize=10)
plt.ylabel('Counts', fontsize=10)

# Distribution of the variable 'age'
ax = plt.figure(figsize=(30, 8))
sns.countplot(train.Age)
axis_font = {'fontname': 'Arial', 'size':'24'}
plt.xlabel('age', **axis_font)
plt.ylabel('Count', **axis_font)

plt.show()
print('End of program')