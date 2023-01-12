import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#-----------------------------------------------------------------------------------------------------

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
        return x * (y - self.sigmoid(np.dot(w.T,x)+b)) - ((alpha*w*w)/N)

    def gradientB(self, x,y,w,b):
        return y - self.sigmoid(np.dot(w.T,x) + b)


    def train(self, Xdata, Ydata, lr, reg, its, Xvalid, Yvalid):
        N = len(Xdata[0])
        obs = len(Ydata)
        W, b = self.initializeParams(N)
        lossValues = []
        accuracyValues = []

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
            
            accuracy = self.validation(W, b, Xvalid, Yvalid)
            accuracyValues.append(accuracy)

            loss = self.loss(Ydata, predictions)
            lossValues.append(loss)

        return W, b, lossValues, accuracyValues
    
    def validation(self, W, b, XData, YData):
        aux = 0
        for i in range(len(YData)):
            z = np.dot(W.T, XData[i]) + b
            value = self.sigmoid(z)
            if(value>0.5):
                pred = 1
            else:
                pred = 0
            
            if(pred == YData[i]):
                aux = aux + 1
        
        accuracy = aux/len(YData) * 100
        return accuracy


#-----------------------------------------------------------------------------------------------------------

data = pd.read_csv('features/features_VGG16.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

s = SelectKBest(k=100)
data_train = data[0:7000].T
X_train = data_train[1:n].T
Y_train = data_train[0]
Y_train = (Y_train != 2).astype(int)
X_train = s.fit_transform(X_train, Y_train)

data_validation = data[7000:m].T
X_valid = data_validation[1:n].T
X_valid = s.transform(X_valid)
Y_valid = data_validation[0]
Y_valid = (Y_valid != 2).astype(int)

learning_rate = 0.001
regularization_param = 0.001
iterations = 10

model = LogisticRegression()
W, b, lossTrain, accuracyValues = model.train(X_train, Y_train, learning_rate, regularization_param, iterations, X_valid, Y_valid)

obs = np.linspace(0,len(lossTrain), num=len(lossTrain))

plt.figure(1)
#plt.plot(obs, lossTrain, label="loss")
plt.plot(obs,accuracyValues, label="accuracy")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.show()
print("The accuracy was {:.2f}%".format(accuracyValues[-1]))
