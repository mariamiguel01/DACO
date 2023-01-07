import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde

def estimate_pdf(data, kernel='gaussian', bandwidth=1.0):
  
  if kernel == 'gaussian':
    kde = gaussian_kde(data.T, bw_method=bandwidth)
  elif kernel == 'tophat':
    kde = gaussian_kde(data.T, bw_method='scott')
  else:
    raise ValueError(f'Invalid kernel: {kernel}')
  def pdf(x):
    return kde.evaluate(x)
  
  return pdf


train_features = pd.read_csv('features_file_resnet.csv')
train_labels = pd.read_csv("train_labels.csv", index_col="id")
train_labels = train_labels[:len(train_features)]
x_train,x_validate,y_train,y_validate=train_test_split(train_features,train_labels,test_size=0.2)


# Estimate the PDF for each class using the Parzen window method
pdfs = []
for c in np.unique(y_train):
  # Select the feature vectors belonging to class c
  X_c = x_train[y_train == c]
  # Estimate the PDF or class c using the Parzen window method
  pdf_c = estimate_pdf(X_c, kernel='gaussian', bandwidth=1.0)
  pdfs.append(pdf_c)

# Classify new feature vectors by comparing the estimated PDFs
predictions = []
for x in y_validate:
  # Compute the estimated PDF for each class at the point x
  pdf_values = [pdf(x) for pdf in pdfs]
  # Choose the class with the highest estimated PDF
  c = np.argmax(pdf_values)
  predictions.append(c)

# Evaluate the accuracy of the classifier
accuracy = np.mean(predictions == y_validate)
print(f'Accuracy: {accuracy:.2f}')
