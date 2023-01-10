import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.stats import gaussian_kde

def kde_sklearn(X_c, x_grid, bandwidth=0.5):
  total_value = 0
  for i in range(len(X_c)):   
    feature_vec = np.array(X_c)
    kde = gaussian_kde(feature_vec.T)
    gaussian = lambda x: kde.evaluate(x)
    gaussian_atpoint=gaussian(X_c)
    value=gaussian(x_grid)
    total_value += value
    print(len(value))
  return total_value


train_features_csv = np.loadtxt('features_file_resnet.csv', delimiter=',')
train_labels = pd.read_csv("train_labels.csv", index_col="id")
species_labels = sorted(train_labels.columns.unique())
class_num=np.size(species_labels)

x = train_features_csv[:, 1:]
y = train_features_csv[:,0]

x_train, x_validate, y_train, y_validate = train_test_split(
    x, y, stratify=y, test_size=0.1
)
predictions = []

for i in tqdm(x_validate):
  p_array = []
  for c in tqdm(range(8)):
      pdfs = []
      X_c = x_train[y_train==c]
      pdf_c = kde_sklearn(X_c,i,bandwidth=0.5)
      print(pdf_c)
      p_array.append(pdf_c)
  pred = np.argmax(np.array(p_array))
  predictions.append(pred)
  

# Evaluate the accuracy of the classifier
accuracy = np.mean(predictions == y_validate)
print(f'Accuracy: {accuracy:.2f}')
