# import necessary libraries
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

###import the data from the train set and subdivide it into train and validation; randomize the data 
data = pd.read_csv('features/features_VGG16_train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
data = data.T
s = StandardScaler()
x = data[2:n].T
x = s.fit_transform(x)
y = data[1]
x_train, x_validate, y_train, y_validate = train_test_split(
    x, y, stratify=y, test_size=0.20
)

x_train=np.array(x_train)
x_validate=np.array(x_validate)
y_train=np.array(y_train)
y_validate=np.array(y_validate)
y_train = y_train[0:7599]
y_validate = y_validate[0:1900]
y_train=y_train.astype(int)
y_validate=y_validate.astype(int)

###Finding the best c value for the train set and the validation set
clf = svm.SVC(kernel='sigmoid')
C_range = [0.2,0.4,0.6,0.8,1, 2,5, 10, 100, 1000]
param_grid = dict(C=C_range)
grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
grid.fit(x_train, y_train)
predictions = {C: grid.predict(x_validate) for C in C_range}
accuracies = {C: accuracy_score(y_validate, predictions[C]) for C in C_range}
best_C = max(accuracies, key=accuracies.get)
print("best C:",grid.best_params_["C"])
print("Best C value validation: ", best_C)

###Importing the test set and randomize it
test = pd.read_csv('features/features_VGG16_test.csv')
test = np.array(test)
m, n = test.shape
np.random.shuffle(test)
test = test.T
s = StandardScaler()
x = test[2:n].T
x_test = s.fit_transform(x)
y_test = test[1] 
x_test=np.array(x_test)
y_test=np.array(y_test)
y_test = y_test[0:500]
y_test=y_test.astype(int)

###creating an svm classifier(OneVsRestClassifier) with a sigmoid kernel and the best c value,and evaluating its performance on the test set
clf = svm.SVC(kernel='sigmoid', C=best_C)
clf = OneVsRestClassifier(clf)
clf.fit(x_train, y_train)
accuracy = clf.score(x_validate,y_validate)
accuracy_test=clf.score(x_test,y_test)
print('Accuracy(%):',accuracy)
print('Accuracy(%) of test :',accuracy_test)
predictions = clf.predict(x_validate)
predictions_2  =clf.predict(x_test)


###performing the confusion matrix
confMatrix = metrics.confusion_matrix(y_test, predictions_2, normalize = None)
display = metrics.ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot()
plt.show()
plt.title('Confusion Matrix - SVM')

