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

##IMPORTING THE TRAIN DATASET
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
print(len(x_train))
print(y_train)
print(len(x_validate))
print(y_validate)

###FINDING THE BEST C VALUE FOR THE TRAIN DATA
# create an SVM classifier
clf = svm.SVC(kernel='sigmoid')
# define the range of C values to test
C_range = [0.2,0.4,0.6,0.8,1, 2,5, 10, 100, 1000]
# create a dictionary of parameters to test
param_grid = dict(C=C_range)
# create a GridSearchCV object
grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
# fit the GridSearchCV object to the data
grid.fit(x_train, y_train)
grid1=grid.score(x_validate,y_validate)
scores = grid.cv_results_['mean_test_score']
# find the index of the highest mean test score
best_index = np.argmax(scores)
# print the best C value and its accuracy on validation set
print("Best C value: ", grid.best_params_['C'])
print("Best accuracy on validation set: ", scores[best_index])

###IMPORTIG THE DATASET OF TEST
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

##TRAINING THE MODEL AND TESTIG IT
# create SVM classifier
clf = svm.SVC(kernel='sigmoid', C=scores[best_index])
# wrap classifier in OneVsRestClassifier
clf = OneVsRestClassifier(clf)

# train classifier on training data
clf.fit(x_train, y_train)
# evaluate classifier on test data
accuracy = clf.score(x_validate,y_validate)
accuracy_test=clf.score(x_test,y_test)
print('Accuracy(%):',accuracy)
print('Accuracy(%) of test :',accuracy_test)

# generate predictions on the validation set
predictions = clf.predict(x_validate)
predictions_2  =clf.predict(x_test)
# generate classification report
report = classification_report(y_validate, predictions)
report=classification_report(y_test,predictions_2)
print(report)

confMatrix = metrics.confusion_matrix(y_test, predictions_2, normalize = None)
display = metrics.ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot()
plt.show()
plt.title('Confusion Matrix - LogisticRegression')

