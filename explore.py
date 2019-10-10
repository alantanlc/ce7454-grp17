import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import argparse

'exec(%matplotlib inline)'

train = pd.read_csv('data/CheXpert-v1.0-small/train.csv')
test = pd.read_csv('data/CheXpert-v1.0-small/valid.csv')

print(test.head())

# Distribution of the target variable
detected_counts = train.Fracture.value_counts()
plt.figure(figsize = (12,6))
sns.barplot(detected_counts.index, detected_counts.values, alpha = 0.9)
plt.xticks(rotation = 'vertical')
plt.xlabel('Fracture labels', fontsize = 12)
plt.ylabel('Counts', fontsize = 12)
plt.show()

# Distribution of the variable 'age'
ax = plt.figure(figsize=(30, 8))
sns.countplot(train.Age)
axis_font = {'fontname': 'Arial', 'size':'24'}
plt.xlabel('age', **axis_font)
plt.ylabel('Count', **axis_font)
plt.show()