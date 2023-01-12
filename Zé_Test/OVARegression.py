import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics

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

data = pd.read_csv('features/features_VGG16.csv')

K = 8
output_classes = ['0', '1', '2', '3', '4', '5', '6', '7']
output_classes = np.concatenate((output_classes, [output_classes[0]]))
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

s = StandardScaler()
data_train = data[0:16000].T
X_train_raw = data_train[1:n].T
X_train = s.fit_transform(X_train_raw)
Y_train = data_train[0]

data_validation = data[16000:m].T
X_valid_raw = data_validation[1:n].T
X_valid = s.transform(X_valid_raw)
Y_valid = data_validation[0]

learning_rate = 0.001
regularization_param = 0.001
iterations = 10


model = OneVersusAll()
W, b, accuracy, Ypred = model.train(X_train, Y_train, learning_rate, regularization_param, iterations, X_valid, Y_valid, K)

Ypred = np.array(Ypred).astype(int)
Y_valid = Y_valid.astype(int)
#obs = np.linspace(0,len(AccK[0]), num=len(AccK[0]))

'''
for i in range(len(AccK)):
    plt.plot(obs,AccK[i], label="accuracy class " + str(i))
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.show() 
'''

confMatrix = metrics.confusion_matrix(Y_valid, Ypred, normalize = None)
display = metrics.ConfusionMatrixDisplay(confusion_matrix = confMatrix)
display.plot()
plt.show()
plt.title('Confusion Matrix - LogisticRegression')

print("The accuracy was {:.2f}%".format(accuracy))