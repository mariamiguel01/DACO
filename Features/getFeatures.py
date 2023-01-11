from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import preprocess_input
from keras.utils import img_to_array
from keras.utils import load_img
from imutils import paths
import numpy as np
import pickle
import random
from imutils import paths
from keras.applications import VGG16
from keras.applications import ResNet101, ResNet50
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


model = VGG16(weights="imagenet", include_top=True, input_shape=(224,224,3))
le = None

imagePaths = list(paths.list_images("images/train"))
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-2] for p in imagePaths]

if le is None:
	le = LabelEncoder()
	le.fit(labels)

csvPath = "features/features_file_VGG16.csv"
csv = open(csvPath, "w")

for (b,i) in enumerate(range(0,len(imagePaths), BATCH_SIZE)):
    batchPaths = imagePaths[i:i + BATCH_SIZE]
    batchLabels = le.transform(labels[i:i + BATCH_SIZE])
    batchImages = []
    
    for imagePath in batchPaths:
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        batchImages.append(image)
    
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=BATCH_SIZE)
    features = features.reshape((features.shape[0], 1000))
    for (label, vec) in zip(batchLabels, features):
        vec = ",".join([str(v) for v in vec])
        csv.write("{},{}\n".format(label, vec))

csv.close()
f = open(LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()