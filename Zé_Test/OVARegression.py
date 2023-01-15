'''
OVARegression 

This file was developed as a project for DACO subject from Bioengeneering Masters at FEUP

It preforms an One Verus All Logistic Regression to classify an image as having or present one of 7 species of wild animals.
It is tested for different feature extraction methods used for transder learning.

The features are read from .csv files
'''



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import f1_score


# This class preforms a Logistic Regression, using the gradient descent method and the equations presented in the report
class LogisticRegression():

    def initializeParams(self, N):
        weighs = np.zeros(N)
        b = 0
        return weighs, b

    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))
    
    def loss(self, y_true, y_pred):
        loss = 0
        for i in range(len(y_true)):
            l = -1 * y_true[i] * np.log10(y_pred[i]) + (1-y_true[i]) * np.log10(1-y_pred[i])
            loss = loss + l
        return loss
    
    def gradientW(self, x, y, w, b, alpha, N):
        return x * (y - self.sigmoid(np.dot(w.T,x)+b)) - ((2*alpha*w)/N)

    def gradientB(self, x,y,w,b):
        return y - self.sigmoid(np.dot(w.T,x) + b)


    def train(self, Xdata, Ydata, lr, reg, its):
        N = len(Xdata[0])
        obs = len(Ydata)
        W, b = self.initializeParams(N)
        lossValues = []

        for i in tqdm(range(0, its)):
            for j in range(obs):
                dw = self.gradientW(Xdata[j], Ydata[j], W, b, reg, N)
                db = self.gradientB(Xdata[j], Ydata[j], W, b)
                W = np.array(W) + lr * np.array(dw)
                b = b + lr * db


            predictions = []
            for j in range(obs):
                z = np.dot(W.T, Xdata[j]) + b
                predictions.append(self.sigmoid(z))
        

            loss = self.loss(Ydata, predictions)
            lossValues.append(loss)

        return W, b, lossValues


# This class uses the previous one to compute 8 different Logistic Regressions, one for each class available.
class OneVersusAll():
    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))

    def validationOVA(self, W, b, XData, YData, K):
        aux = 0
        Ypred = []
        for i in range(len(YData)):
            pValues = []
            for k in range(K): 
                z = np.dot(W[k].T, XData[i]) + b[k]
                value = self.sigmoid(z)
                pValues.append(value)
                
            pred = np.argmax(np.array(pValues))
            Ypred.append(pred)
            if(pred == YData[i]):
                aux = aux + 1
            
        accuracy = aux/len(YData) * 100
        return accuracy, Ypred
    
    def train(self, Xdata, Ydata, lr, reg, its, Xvalid, Yvalid, K):
        W = []
        b = []
        for i in range(K):
            Ydata_aux  = (Ydata == i).astype(int)
            model = LogisticRegression()
            Wi, bi, lossValues = model.train(Xdata, Ydata_aux, lr, reg, its)
            W.append(Wi)
            b.append(bi)
        
        accuracy, Ypred = self.validationOVA(W, b, Xvalid, Yvalid, K)

        return W, b, accuracy, Ypred

# This are the train files used, one for each extraction method
files = ['features/features_DenseNet121_train.csv','features/features_ResNet50_train.csv','features/features_VGG16_train.csv']
#  This are the test files used, one for each extraction method
test_files = ['features/features_DenseNet121_test.csv', 'features/features_ResNet50_test.csv', 'features/features_VGG16_test.csv']
modelsNames = ['DenseNet121', 'ResNet50', 'VGG16']

# This value should be True to test the model and False to preform validations. If False, the program does not read the test .csv and devides the
# train .csv into train and validtion
test = True

# This are the parameters used in the model. To test different regularizations and learning rates, the corresponding variable should be changed to
# an array containing the values that need to be tested
K = 8
learning_rate = 0.001
regularization_param =  0.001
iterations = 5

accuracyModels = []
lossModels = []
accuracyTest = []
f1Models = []


for i in range(len(files)):
    data = pd.read_csv(files[i])

    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    s = StandardScaler() #It is used to normalize both the train and test features

    if test: #Reads the test file
        data_test = pd.read_csv(test_files[i])
        data_test = np.array(data_test)
        m, n = data_test.shape
        np.random.shuffle(data_test)

        data_train = data.T
        X_train = data_train[2:n].T
        Y_train = data_train[1]
        X_train = s.fit_transform(X_train)

        data_test = data_test.T
        X_valid = data_test[2:n].T
        X_valid = s.transform(X_valid)
        Y_valid = data_test[1]


    else: #Divides the train file into test and validation
        data_train = data[0:9000].T
        X_train = data_train[2:n].T
        Y_train = data_train[1]
        X_train = s.fit_transform(X_train)

        data_validation = data[9000:m].T
        X_valid = data_validation[2:n].T
        X_valid = s.transform(X_valid)
        Y_valid = data_validation[1]

    model = OneVersusAll()
    accuracyValues = []
    
    '''
    # To test different learning rates, uncomment this cicle 
    for lr in learning_rate:
        W, b, accuracy, Ypred = model.train(X_train, Y_train, lr, regularization_param, iterations, X_valid, Y_valid, K)
        accuracyValues.append(accuracy)
    accuracyModels.append(accuracyValues)
    
    # To test different regularization values, uncomment this cicle
    for rp in regularization_param:
        W, b, accuracy, Ypred = model.train(X_train, Y_train, learning_rate, rp, iterations, X_valid, Y_valid, K)
        accuracyValues.append(accuracy)
    accuracyModels.append(accuracyValues)
    '''
    
    # The following lines are used to test the model for each extraction method
    if test:
        W, b, accuracy, Ypred = model.train(X_train, Y_train, learning_rate, regularization_param, iterations, X_valid, Y_valid, K)
        accuracyTest.append(accuracy)
        Y_valid = Y_valid.astype(int)
        Ypred = np.array(Ypred).astype(int)
        f1 = f1_score(Y_valid, Ypred, average='weighted') * 100
        f1Models.append(f1)

    '''
    # To plot a confusion matrix of the test result, uncomment the following lines
    confMatrix = metrics.confusion_matrix(Y_valid, Ypred, normalize = None)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix = confMatrix)
    display.plot()
    plt.show()
    '''
    

if test: # Prints the accuracy and F1 score values when testing
    for i in range(len(accuracyTest)):
        print(modelsNames[i] + ": " + "Acc: {:.2f}% | F1: {:.2f}%".format(accuracyTest[i], f1Models[i]))
else: # The following lines are used to plot the validation values
    i=0
    for ac in accuracyModels:
        for j in range(len(learning_rate)):
            print("Model: " + str(modelsNames[i]) + " RP: " + str(learning_rate[j]) + " acc: " + str(ac[j]))
        i += 1

    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy for different Learning Rates")
    plt.ylim(ymax = 95, ymin = 80)
    plt.show()




