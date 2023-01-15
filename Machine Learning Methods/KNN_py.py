import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

train_features_csv = np.loadtxt('features_file_resnet.csv', delimiter=',')

x = train_features_csv[:, 1:]
y = train_features_csv[:,0]

x_train, x_eval, y_train, y_eval = train_test_split(
    x, y, stratify=y, test_size=0.25 #TESTAR o stratify
)

k_test = [12,15,18,25]

for K in k_test:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(x_train, y_train)
    predictions = []
    for sample in tqdm(x_eval):
        xx= knn.predict([sample])
        predictions.append(xx)
    accuracy = np.mean(predictions == y_eval)*100
    print(K, f' Accuracy: {accuracy:.2f}')