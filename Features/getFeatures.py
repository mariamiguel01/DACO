'''
Get Features

This file was developed as a project for DACO subject from Bioengeneering Masters at FEUP

It runs a feature extraction model in a set of images.

The features are saved in .csv files
'''


from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import preprocess_input
from keras.utils import img_to_array
from keras.utils import load_img
import numpy as np
import pickle
import random
from imutils import paths
from keras.applications import VGG16, ResNet50, DenseNet121
import keras as K
import os
import numpy as np

TRAIN = "train_features"
TEST = "test_features"
BATCH_SIZE = 32
BASE_CSV_PATH = "features"
MODEL_PATH = "features/model.cpickle"
LE_PATH = "features/le_resnet.cpickle"
CLASSES = ['antelope_duiker',
 'bird',
 'blank',
 'civet_genet',
 'hog',
 'leopard',
 'monkey_prosimian',
 'rodent']

# This is model used. To extract features from each three, change from ResNet50 to VVG16 and DenseNet121
res_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))

# When using Resnet50 or DenseNet121, use this cicle to stop the process. The value should be 143 for ResNet50 and 149 for DenseNet121
for layer in res_model.layers[:143]:
    layer.trainable = False

# This adds the MaxPooling operation and a Flatten opertion to the model
model = K.models.Sequential()
model.add(res_model)
model.add(K.layers.MaxPooling2D(pool_size=(7, 7),strides=(2, 2), padding='valid'))
model.add(K.layers.Flatten())
model.summary()
le = None

# The model is then applied to each image in the dataset using batches of 32
imagePaths = list(paths.list_images("images/train"))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
namesFile = [p.split(os.path.sep)[-1] for p in imagePaths]
names = [p.split(".")[0] for p in namesFile]

if le is None:
	le = LabelEncoder()
	le.fit(labels)

csvPath = "features/features_DenseNet121.csv"
csv = open(csvPath, "w")

for (b,i) in enumerate(range(0,len(imagePaths), BATCH_SIZE)):
    batchPaths = imagePaths[i:i + BATCH_SIZE]
    batchLabels = le.transform(labels[i:i + BATCH_SIZE])
    batchNames = names[i:i + BATCH_SIZE]
    batchImages = []
    
    for imagePath in batchPaths:
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        batchImages.append(image)
    
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=BATCH_SIZE)
    features = features.reshape((features.shape[0], 2048)) #The second value should be 2048 for ResNet50, 1024 for DenseNet121 and 512 for VGG16
    for (imageId, label, vec) in zip(batchNames, batchLabels, features):
        vec = ",".join([str(v) for v in vec])
        csv.write("{},{},{}\n".format(imageId, label, vec))

csv.close()
f = open(LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()
