import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt


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


x_train, x_validate, y_train, y_validate = train_test_split(
    x, y, stratify=y, test_size=0.01
)


predictions = []
p_array=[]
pdfs=[]

#for i in tqdm(range(x_validate.shape[0])):
for i in tqdm(range(x_test.shape(0)))
  pdfs = []
  for c in range(7):
      X_c = x_train[y_train==c]
      #x = x_validate[i]
      x= x_test[i]
      pdf_c = evaluate_gaussian_all(X_c, x)
      pdfs.append(pdf_c)
  pred=np.argmax(pdfs)
  predictions.append(pred)

print(predictions)
# Evaluate the accuracy of the classifier
#accuracy = np.mean(predictions == y_validate)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy:.2f}')

confMatrix = metrics.confusion_matrix(y_test, predictions, normalize = None)
display = metrics.ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot()
plt.show()
plt.title('Confusion Matrix - LogisticRegression')