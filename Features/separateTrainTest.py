from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data_vgg16 = pd.read_csv("features/features_VGG16.csv")
data_vgg16 = np.array(data_vgg16)
m,n = data_vgg16.shape
np.random.shuffle(data_vgg16)

data_vgg16 = data_vgg16[0:10000].T
x_vgg16 = data_vgg16[2:n].T
y_vgg16 = data_vgg16[1]
names = data_vgg16[0]

xVG_train, xVG_test, yVG_train, yVG_test, namesVG_train, namesVG_test = train_test_split(
    x_vgg16, y_vgg16, names, stratify=y_vgg16, test_size=0.05
)


csvPathVG_train = "features/features_VGG16_train.csv"
csv1 = open(csvPathVG_train, "w")

for (imageId, label, vec) in zip(namesVG_train, yVG_train, xVG_train):
    vec = ",".join([str(v) for v in vec])
    csv1.write("{},{},{}\n".format(imageId, label, vec))

csv1.close()

csvPathVG_test = "features/features_VGG16_test.csv"
csv2 = open(csvPathVG_test, "w")

for (imageId, label, vec) in zip(namesVG_test, yVG_test, xVG_test):
    vec = ",".join([str(v) for v in vec])
    csv2.write("{},{},{}\n".format(imageId, label, vec))

csv2.close()

data_resnet = pd.read_csv("features/features_ResNet50.csv")
m1,n1 = data_resnet.shape
data_resnet = np.array(data_resnet).T
namesRes = data_resnet[0]
labelsRes = data_resnet[1]
valuesRes = data_resnet[2:n1].T

interNamesRes_train = np.in1d(namesRes, namesVG_train)
interNamesRes_test = np.in1d(namesRes, namesVG_test)

data_resnet = data_resnet.T
dataRes_train = data_resnet[interNamesRes_train].T
dataRes_test = data_resnet[interNamesRes_test].T


xRes_train = dataRes_train[2:n1].T
xRes_test = dataRes_test[2:n1].T
yRes_train = dataRes_train[1]
yRes_test = dataRes_test[1]
namesRes_train = dataRes_train[0]
namesRes_test = dataRes_test[0]


csvPathRes_test = "features/features_ResNet50_test.csv"
csv3 = open(csvPathRes_test, "w")

for (imageId, label, vec) in zip(namesRes_test, yRes_test, xRes_test):
    vec = ",".join([str(v) for v in vec])
    csv3.write("{},{},{}\n".format(imageId, label, vec))

csv3.close()

csvPathRes_train = "features/features_ResNet50_train.csv"
csv4 = open(csvPathRes_train, "w")

for (imageId, label, vec) in zip(namesRes_train, yRes_train, xRes_train):
    vec = ",".join([str(v) for v in vec])
    csv4.write("{},{},{}\n".format(imageId, label, vec))

csv4.close()


#-------

data_dense = pd.read_csv("features/features_DenseNet121.csv")
m2,n2 = data_dense.shape
data_dense = np.array(data_dense).T
namesDense = data_dense[0]


interNamesDense_train = np.in1d(namesDense, namesVG_train)
interNamesDense_test = np.in1d(namesDense, namesVG_test)

data_dense = data_dense.T
dataDense_train = data_dense[interNamesDense_train].T
dataDense_test = data_dense[interNamesDense_test].T


xDense_train = dataDense_train[2:n2].T
xDense_test = dataDense_test[2:n2].T
yDense_train = dataDense_train[1]
yDense_test = dataDense_test[1]
namesDense_train = dataDense_train[0]
namesDense_test = dataDense_test[0]


csvPathDense_test = "features/features_DenseNet121_test.csv"
csv5 = open(csvPathDense_test, "w")

for (imageId, label, vec) in zip(namesDense_test, yDense_test, xDense_test):
    vec = ",".join([str(v) for v in vec])
    csv5.write("{},{},{}\n".format(imageId, label, vec))

csv5.close()

csvPathDense_train = "features/features_DenseNet121_train.csv"
csv6 = open(csvPathDense_train, "w")

for (imageId, label, vec) in zip(namesDense_train, yDense_train, xDense_train):
    vec = ",".join([str(v) for v in vec])
    csv6.write("{},{},{}\n".format(imageId, label, vec))

csv6.close()