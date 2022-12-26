import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import torchvision.models as models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image

train_features = pd.read_csv("train_features.csv", index_col="id")
#test_features = pd.read_csv("test_features.csv", index_col="id")
train_labels = pd.read_csv("train_labels.csv", index_col="id")

# frac = 0.5

# y = train_labels.sample(frac=frac, random_state=1)
# x = train_features.loc[y.index].filepath.to_frame()

# # note that we are casting the species labels to an indicator/dummy matrix
# x_train, x_eval, y_train, y_eval = train_test_split(
#     x, y, stratify=y, test_size=0.25
# )


# split_pcts = pd.DataFrame(
#     {
#         "train": y_train.idxmax(axis=1).value_counts(normalize=True),
#         "eval": y_eval.idxmax(axis=1).value_counts(normalize=True),
#     }
# )
# print("Species percentages by split")
# (split_pcts.fillna(0) * 100).astype(int)

img_path = 'test_features/ZJ016488.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
model = VGG16(weights='imagenet', include_top=False)
features = model.predict(x)

print(x)