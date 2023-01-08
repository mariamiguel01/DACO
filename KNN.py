import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split


train_features_csv = np.loadtxt('features_file_resnet.csv', delimiter=',')
train_labels = pd.read_csv("train_labels.csv", index_col="id")
species_labels = sorted(train_labels.columns.unique())

x = train_features_csv[:, 1:]
y = train_features_csv[:,0]

x_train, x_eval, y_train, y_eval = train_test_split(
    x, y, stratify=y, test_size=0.25 #TESTAR o stratify
)

#Step-1: Select the number K of the neighbors
K = 5
#Step-2: Calculate the Euclidean distance of K number of neighbors
all_distances = []
all_prob = []
predictions = []
for sample in x_eval:
    distance = []
    for x in x_train:
        distance.append(sum((sample - x)**2))
    sorted_distances_index = np.argsort(distance)[::-1][:K]
    all_distances.append(sorted_distances_index)

    sample_prob = []
    for category in range(8):
        sample_prob.append(sum(y_train[sorted_distances_index] == category)/K)
    all_prob.append(sample_prob)

    predictions.append(np.argmax(sample_prob))

#Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.
#Step-4: Among these k neighbors, count the number of the data points in each category.
#Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.
#Step-6: Our model is ready.

# Evaluate the accuracy of the classifier
accuracy = np.mean(predictions == y_eval)*100
print(f'Accuracy: {accuracy:.2f}')