import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

# import feature files
train_samples_csv = pd.read_csv('Features/features_ResNet50_train.csv', delimiter=',')
test_samples_csv = pd.read_csv('Features/features_ResNet50_test.csv', delimiter=',')

train_samples = np.array(train_samples_csv)
x_train_ = train_samples[:, 2:]
y_train_ = train_samples[:,1]

# Split the training samples in train and validation sets 
x_train, x_eval, y_train, y_eval = train_test_split(
    x_train_, y_train_, stratify=y_train_, test_size=0.20
)

# Test samples
""" test_samples = np.array(test_samples_csv)
x_test = test_samples[:, 2:]
y_test = test_samples[:,1] """

# KNN algorithm
#1: Select the number K of the neighbors
k_test = [5,8,10] #used to train and validation
#k_test = [5] #used to test

for K in k_test:
    #2: Calculate the Euclidean distances
    all_distances = []
    all_prob = []
    predictions = []
    for sample in tqdm(x_eval):
        distance = []
        for x in x_train:
            distance.append(sum((sample - x)**2))

        #3: Take the K nearest neighbours to evaluate the volume
        sorted_distances_index = np.argsort(distance)[:K]
        all_distances.append(sorted_distances_index)

        #4: Among these K neighbours, count the number of the data points in each category
        sample_prob = []
        for category in range(8):
            sample_prob.append(sum(y_train[sorted_distances_index] == category)/K)
        all_prob.append(sample_prob)

        #5: Assign the new data points to the category for which the number of the neighbourhood is maximum
        predictions.append(np.argmax(sample_prob))

    # Evaluate the accuracy of the classifier
    accuracy = np.mean(predictions == y_eval)*100
    print(K, f' Accuracy: {accuracy:.2f}')

# Code used only for the test:
""" Ypred = np.array(predictions).astype(int)
Yvalid = y_test.astype(int)

# Plot the confusion matrix
confMatrix = metrics.confusion_matrix(Yvalid, Ypred, normalize = None)
display = metrics.ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot()
plt.show()
plt.title('Confusion Matrix') """