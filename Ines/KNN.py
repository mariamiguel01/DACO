import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

train_samples_csv = pd.read_csv('Features/features_ResNet50_train.csv', delimiter=',')
test_samples_csv = pd.read_csv('Features/features_ResNet50_test.csv', delimiter=',')

train_samples = np.array(train_samples_csv)
x_train_ = train_samples[:, 2:]
y_train_ = train_samples[:,1]

# Validation 

""" x_train, x_eval, y_train, y_eval = train_test_split(
    x_train_, y_train_, stratify=y_train_, test_size=0.20
) """

# Test
test_samples = np.array(test_samples_csv)
x_test = test_samples[:, 2:]
y_test = test_samples[:,1]

#1: Select the number K of the neighbors
#k_test = [5,8,10]
k_test = [5]

for K in k_test:
    #2: Calculate the Euclidean distance of K number of neighbors
    all_distances = []
    all_prob = []
    predictions = []
    for sample in tqdm(x_test):
        distance = []
        for x in x_train_:
            distance.append(sum((sample - x)**2))

        #3: Take the K nearest neighbors to calculate the Euclidean distance
        sorted_distances_index = np.argsort(distance)[:K]
        all_distances.append(sorted_distances_index)

        #4: Among these K neighbors, count the number of the data points in each category
        sample_prob = []
        for category in range(8):
            sample_prob.append(sum(y_train_[sorted_distances_index] == category)/K)
        all_prob.append(sample_prob)

        #5: Assign the new data points to the category for which the number of the neighbor is maximum
        predictions.append(np.argmax(sample_prob))

    # Evaluate the accuracy of the classifier
    accuracy = np.mean(predictions == y_test)*100
    print(K, f' Accuracy: {accuracy:.2f}')


Ypred = np.array(predictions).astype(int)
Yvalid = y_test.astype(int)

confMatrix = metrics.confusion_matrix(Yvalid, Ypred, normalize = None)
display = metrics.ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot()
plt.show()
plt.title('Confusion Matrix - densenet')