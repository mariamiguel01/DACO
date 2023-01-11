import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.stats import gaussian_kde
from typing import List
from scipy.stats import multivariate_normal


def evaluate_gaussian_all(X_c: np.ndarray, x: np.ndarray)->np.ndarray:
  gaussian_list = []
  totalvalue=0
  for i in range(X_c.shape[0]):
      mu = X_c[i]
      sigma = np.identity(X_c.shape[1])
      gaussian = lambda x :multivariate_normal(mu, sigma)
      gaussian_list.append(gaussian(x))
  return np.array(gaussian_list).T

train_features_csv = np.loadtxt('features_file_VGG16.csv', delimiter=',')
train_labels = pd.read_csv("train_labels.csv", index_col="id")
species_labels = sorted(train_labels.columns.unique())
class_num=np.size(species_labels)

x = train_features_csv[:, 1:]
y = train_features_csv[:,0]


x_train, x_validate, y_train, y_validate = train_test_split(
    x, y, stratify=y, test_size=0.1
)
predictions = []
p_array=[]
pdfs=[]
for i in range(x_validate.shape[0]):
  pdfs = []
  for c in tqdm(range(7)):
      X_c = x_train[y_train==c]
      x = x_validate[i]
      pdf_c = evaluate_gaussian_all(X_c, x)
      pdfs.append(pdf_c)
  pred=np.argmax(pdfs)
  predictions.append(pred)

# Evaluate the accuracy of the classifier
accuracy = np.mean(predictions == y_validate)
print(f'Accuracy: {accuracy:.2f}')
