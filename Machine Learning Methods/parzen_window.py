### import necessary libraries
#the code that is commented should be uncommented if we want to validate the sigma value by testing different ones
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

###defining a gaussian function that will create a gaussian for each point of the train set with that class and evaluate de probability of the test set in that gaussian and sum them for all points belonging to a class
def evaluate_gaussian_all(X_c, x):
  gaussian_list = []
  totalvalue=0
  n = len(x)
  for i in range(X_c.shape[0]):
      mu = X_c[i]
      sigma = np.identity(n)*0.825
      prob = np.exp(-0.5*(x-mu).T @ np.linalg.inv(sigma) @ (x-mu))
      totalvalue += prob
  return totalvalue

###import the data from the train set and subdivide it into train and validation; randomize the data 
train_labels = pd.read_csv("train_labels.csv", index_col="id")
species_labels = sorted(train_labels.columns.unique())
class_num=np.size(species_labels)

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
    x, y, stratify=y, test_size=0.01
)
###limit the data to 3000 of training because it is a very slow method
x_train=x_train[1:3000]
y_train=y_train[1:3000]
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
###limit the data to 300 of test because it is a very slow method
y_test = y_test[0:300]
y_test=y_test.astype(int)

predictions = []
p_array=[]
pdfs=[]
###evaluating for each point of the test set and for each class rthe probability of the test point belonging to a gaussian of that class, it also chooses the best class for each point by maximizing the prediction value
#for i in tqdm(range(x_validate.shape[0])):
for i in tqdm(range(x_test.shape[0])):
  pdfs = []
  for c in range(7):
      X_c = x_train[y_train==c]
      #x = x_validate[i]
      x= x_test[i]
      pdf_c = evaluate_gaussian_all(X_c, x)
      pdfs.append(pdf_c)
  pred=np.argmax(pdfs)
  predictions.append(pred)


### Evaluate the accuracy of the classifier
#accuracy = np.mean(predictions == y_validate)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy:.2f}')

###computing the confusion matrix
confMatrix = metrics.confusion_matrix(y_test, predictions, normalize = None)
display = metrics.ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot()
plt.show()
plt.title('Confusion Matrix - Parzen Window')