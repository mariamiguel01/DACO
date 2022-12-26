from imutils import paths
from keras.applications import ResNet50
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import shutil
import torchvision.models as models

train_features = pd.read_csv("train_features.csv", index_col="id")
train_labels = pd.read_csv("train_labels.csv", index_col="id")

frac = 1000/len(train_features)
y = train_labels.sample(frac=frac, random_state=1)
x = train_features.loc[y.index].filepath.to_frame()

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
        image = Image.open(self.data.iloc[index]["filepath"]).convert("RGB")
        image = self.transform(image)
        image_id = self.data.index[index]
        # if we don't have labels (e.g. for test set) just return the image and image id
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

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

csvPath = "features/features_file_resnet.csv"
csv = open(csvPath, "w")
aux = 0

for batch_n, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    batchLabels = []
    aux = aux + len(batch)
    
    for i in range(train_dataloader.batch_size):
        batchLabels.append(np.where(batch['label'][0].numpy() == 1)[0][0])
    
    features = model(batch["image"]).detach().numpy()
    features = features.reshape((features.shape[0], 1000))
    aux = aux + len(batchLabels)

    imageNames = batch["image_id"]
    
    for (imageId, label, vec) in zip(imageNames, batchLabels, features):
        vec = ",".join([str(v) for v in vec])
        csv.write("{},{},{}\n".format(imageId, label, vec))

csv.close()
print(aux)