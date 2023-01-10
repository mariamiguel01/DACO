# import necessary libraries
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

train_features_csv = np.loadtxt("features_file.csv", delimiter=",")
train_labels = pd.read_csv("train_labels.csv", index_col="id")


x = train_features_csv[:, 1:]
y = train_features_csv[:,0]

x_train, x_validate, y_train, y_validate = train_test_split(
    x, y, stratify=y, test_size=0.25 #TESTAR o stratify
)

# create SVM classifier
clf = svm.SVC(kernel='sigmoid', C=2) ##linear poly rbf sigmoid nao sei bem como escolher
# wrap classifier in OneVsRestClassifier
clf = OneVsRestClassifier(clf)

# train classifier on training data
clf.fit(x_train, y_train)
# evaluate classifier on test data
accuracy = clf.score(x_validate,y_validate)
acc=accuracy*100
print('Accuracy(%):', acc)

# generate predictions on the validation set
predictions = clf.predict(x_validate)

# generate classification report
report = classification_report(y_validate, predictions)
print(report)
