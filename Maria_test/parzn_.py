import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.stats import gaussian_kde

def kde_sklearn(X_c, x_validate, bandwidth=0.5):
  kde = KernelDensity(kernel='gaussian',bandwidth=bandwidth).fit(X_c)
  return kde.score_samples(x_validate)


train_features_csv = np.loadtxt('features_file_resnet.csv', delimiter=',')
train_labels = pd.read_csv("train_labels.csv", index_col="id")
species_labels = sorted(train_labels.columns.unique())
class_num=np.size(species_labels)

x = train_features_csv[:, 1:]
y = train_features_csv[:,0]

x_train, x_validate, y_train, y_validate = train_test_split(
    x, y, stratify=y, test_size=0.1
)
print(y_train)
predictions = []

for i in tqdm(x_validate):
  p_array = []
  for c in tqdm(range(7)):
      pdfs = []
      X_c = x_train[y_train==c]
      if(len(X_c)==0):
        pdf_c=0;
      else:
        feature_vec = np.array(X_c)
        validation=np.array(x_validate)
        pdf_c = kde_sklearn(feature_vec,validation,bandwidth=0.5)
      p_array.append(pdf_c)
  pred = np.max(p_array)
  predictions.append(pred)
  

# Evaluate the accuracy of the classifier
accuracy = np.mean(predictions == y_validate)
print(f'Accuracy: {accuracy:.2f}')
