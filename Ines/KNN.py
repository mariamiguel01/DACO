import numpy as np
from tqdm import tqdm
import pandas as pd

train_samples_csv = pd.read_csv('features_ResNet50_train.csv', delimiter=',')
test_samples_csv = pd.read_csv('features_ResNet50_test.csv', delimiter=',')

train_samples = np.array(train_samples_csv)
test_samples = np.array(test_samples_csv)

x_train = train_samples[:, 2:]
y_train = train_samples[:,1]
x_test = test_samples[:, 2:]
y_test = test_samples[:,1]

k_test = [12,15,18,25,30]

#Step-1: Select the number K of the neighbors
for K in k_test:
    #Step-2: Calculate the Euclidean distance of K number of neighbors
    all_distances = []
    all_prob = []
    predictions = []
    for sample in tqdm(x_test):
        distance = []
        for x in x_train:
            distance.append(sum((sample - x)**2))
        sorted_distances_index = np.argsort(distance)[:K]
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
    accuracy = np.mean(predictions == y_test)*100
    print(K, f' Accuracy: {accuracy:.2f}')