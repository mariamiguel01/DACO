import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# import feature files
train_samples_csv = pd.read_csv('Features/features_VGG16_train.csv', delimiter=',')
test_samples_csv = pd.read_csv('Features/features_VGG16_test.csv', delimiter=',')
train_labels = pd.read_csv("train_labels.csv", index_col="id")

x = train_samples_csv.iloc[:, 2:]
y = train_samples_csv.iloc[:,1]
x_test = test_samples_csv.iloc[:, 2:]
y_test = test_samples_csv.iloc[:,1]

# standardize train features
scaler = StandardScaler().fit(x)
scaled_train = scaler.transform(x)
# Split the training samples in train and validation sets 
sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
for train_index, valid_index in sss.split(scaled_train, y):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

nb_features = len(train_samples_csv.columns) - 2 # number of features
nb_class = 8

# Neural Network layers
model = Sequential()
model.add(Conv1D(filters=512, kernel_size=3, input_shape=(nb_features,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(nb_class))
model.add(Activation('softmax'))

y_train = np_utils.to_categorical(y_train, nb_class)
y_valid = np_utils.to_categorical(y_valid, nb_class)
y_test = np_utils.to_categorical(y_test, nb_class)

sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

print("Fit model on training data")
nb_epoch = 3
model.fit(X_train, y_train, epochs=nb_epoch, validation_data=(X_valid, y_valid), batch_size=16)

# Evaluate the model on the test set
print("Evaluate on test data")
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)

# Generate predictions to the test set
predictions = model.predict(x_test)

Ytrue_test = np.argmax(np.array(y_test.astype(int)), axis=1)
Ypred = np.argmax(np.array(predictions).astype(int), axis=1)

# Plot the confusion matrix
confMatrix = metrics.confusion_matrix(Ytrue_test, Ypred, normalize = None)
display = metrics.ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot()
plt.show()
plt.title('Confusion Matrix - Neural Network')