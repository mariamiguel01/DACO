import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

train_samples_csv = pd.read_csv('Features/features_VGG16_train.csv', delimiter=',', usecols=[2:])
test_samples_csv = pd.read_csv('Features/features_VGG16_test.csv', delimiter=',')
train_labels = pd.read_csv("train_labels.csv", index_col="id")
#species_labels = sorted(train_labels.columns.unique())

x = train_samples_csv[:, 2:]
y = train_samples_csv[:,1]
x_test = test_samples_csv[:, 2:]
y_test = test_samples_csv[:,1]

# standardize train features
scaler = StandardScaler().fit(x)
scaled_train = scaler.transform(x)
# split train data into train and validation
sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
for train_index, valid_index in sss.split(scaled_train, y):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

nb_features = len(train_samples_csv.columns) - 2 # number of features
nb_class = len(species_labels)

# standardize test features
scaler_test = StandardScaler().fit(x_test)
scaled_test = scaler_test.transform(x_test)

# Keras model with one Convolution1D layer
# unfortunately more number of covnolutional layers, filters and filters lenght 
# don't give better accuracy
model = Sequential()
model.add(Conv1D(filters=512, kernel_size=1, input_shape=(nb_features,1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
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

model.save('my_model_CNN.h5')

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(scaled_test, y_test)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(scaled_test)

f = open("CNN_results.txt", "a")
f.write("Results")
f.write(results)
f.write("Predictions")
f.write(predictions)
f.close()