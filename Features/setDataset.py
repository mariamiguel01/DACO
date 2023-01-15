'''
Set Dataset

This file was developed as a project for DACO subject from Bioengeneering Masters at FEUP

It separates the images from a folder into a respective one according to its class
It helps to better analyse and organize the project
'''

import os
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import shutil

TRAIN = "train_features"
TEST = "test_features"
BATCH_SIZE = 32
BASE_CSV_PATH = "features"
MODEL_PATH = "features/model.cpickle"
LE_PATH = "features/le.cpickle"
CLASSES = ['antelope_duiker',
 'bird',
 'blank',
 'civet_genet',
 'hog',
 'leopard',
 'monkey_prosimian',
 'rodent']

train_features = pd.read_csv("train_features.csv", index_col="id")
test_features = pd.read_csv("test_features.csv", index_col="id")
train_labels = pd.read_csv("train_labels.csv", index_col="id")

frac = 1

y = train_labels.sample(frac=frac, random_state=1)
x = train_features.loc[y.index].filepath.to_frame()

# This class was given in the benchmark.ipynb provided by the competition. It pre-processes the images and gets data from it
# This will be applyed to organize the images into folders
class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_df, y_df=None):
        self.data = x_df
        self.label = y_df
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __getitem__(self, index):
        image = self.data.iloc[index]["filepath"]
        image_id = self.data.index[index]
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, 
                                 dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)

train_dataset = ImagesDataset(x, y)
train_dataloader = DataLoader(train_dataset, batch_size=32)

# Using the information obtained with the provided class (above) we save the images into different class folders
for i in range(len(train_dataset)):
    filename = train_dataset[i]['image_id']
    label = CLASSES[np.where(train_dataset[i]['label'].numpy() == 1.)[0][0]]

    dirPath = "images/train/" + label
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    p = dirPath + '/' + filename +'.jpg'
    shutil.copy2(train_dataset[i]['image'], p)