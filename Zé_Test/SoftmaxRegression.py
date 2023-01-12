import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


#-----------------------------------------------------------------------------------------------------

class LogisticRegression():

    def initializeParams(self, N, K):
        weighs = np.zeros((K,N))
        return weighs

    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))

    def softmax(self, Z):
        aux_array = np.zeros(len(Z))

        for i in range(len(aux_array)):
            aux_array[i] = np.exp(Z[i])
        
        aux_sum = sum(aux_array)
        for i in range(len(aux_array)):
            aux_array[i] = aux_array[i] / aux_sum

        return aux_array
    
    def onehot_enconder(self, Y, K):
        Yohe = []
        for i in range(len(Y)):
            obs = np.zeros(K)
            obs[Y[i]] = 1
            Yohe.append(obs)
        return Yohe

    def class_probability(self, W,X):
        Zarray = []
        for i in range(len(W)):
            Zi = -np.dot(X, W[i].T)
            Zarray.append(Zi)
        
        P = self.softmax(Zarray)
        return P
    
    # Aqui o Y é a matriz onehot encoded (vetores com vários 0 e um 1 no local repestivo à classe certa)
    def loss(self, X, Yohe, W):
        Z = - X @ W.T
        N = X.shape[0]
        W = np.array(W)
        Yohe = np.array(Yohe)
        loss = 1/N * (np.trace(X @ W.T @ Yohe.T) + np.sum(np.log(np.sum(np.exp(Z), axis = 1))))
        return loss

    def gradientW(self, X, Yohe, W, alpha):
        Z = - X @ W.T
        P = self.softmax(Z)
        N = X.shape[0]

        Yohe = np.array([Yohe])
        P = np.array([P])
        X = np.array([X])
        dw = 1/N * ((Yohe - P).T @ X) + 2 * alpha * W
        return dw


    def train(self, Xdata, Ydata, lr, reg, its, Xvalid, Yvalid, K):
        N = len(Xdata[0])
        obs = len(Ydata)
        Yohe = self.onehot_enconder(Ydata, K)
        W = self.initializeParams(N, K)
        lossValues = []
        accuracyValues = []

        for i in tqdm(range(0, its)):
            for j in range(obs):
                dw = self.gradientW(Xdata[j], Yohe[j], W, reg)
                W = np.array(W) - lr * np.array(dw)

            valid_predictions = []

            loss = self.loss(Xdata, Yohe, W)
            lossValues.append(loss)
            
            for i in range(len(Yvalid)):
                probability = self.class_probability(W, Xvalid[i])
                pred = np.argmax(np.array(probability))
                valid_predictions.append(pred)

            valid_predictions = np.array(valid_predictions)
            accuracy = self.validation(valid_predictions, Yvalid)
            accuracyValues.append(accuracy)

        return W, lossValues, accuracyValues
    
    def validation(self, predictions, YData):
        aux = 0
        for i in range(len(YData)):
            if(predictions[i] == YData[i]):
                aux = aux + 1
              
        accuracy = aux/len(YData) * 100
        return accuracy


#-----------------------------------------------------------------------------------------------------------

data = pd.read_csv('features/features_file.csv')

K = 8
data = np.array(data)
data = data[0:8000]
m, n = data.shape
np.random.shuffle(data)

s = StandardScaler()
data_train = data[0:7000].T
X_train_raw = data_train[1:n].T
X_train = s.fit_transform(X_train_raw)
Y_train = data_train[0]
Y_train = Y_train.astype('int')

data_validation = data[7000:m].T
X_valid_raw = data_validation[1:n].T
X_valid = s.transform(X_valid_raw)
Y_valid = data_validation[0]
Y_valid = Y_valid.astype('int')


learning_rate = 0.1
regularization_param = 0.01
iterations = 150

model = LogisticRegression()
W, lossTrain, accuracyValues = model.train(X_train, Y_train, learning_rate, regularization_param, iterations, X_valid, Y_valid, K)

obs = np.linspace(0,len(lossTrain), num=len(lossTrain))

plt.figure(1)
#plt.plot(obs, lossTrain, label="loss")
plt.plot(obs,accuracyValues, label="accuracy")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.show()
print("The accuracy was {:.2f}%".format(accuracyValues[-1]))

'''
clf = SGDClassifier(eta0 = 0.0001, loss="log_loss", alpha=0.0001,
random_state = 15, penalty="l2", tol = 1e-3,
verbose = 2, learning_rate='constant')

clf.fit(X_train, Y_train)
y_pred = clf.predict(X_valid)

print('Accuracy: {:.2f}'.format(accuracy_score(Y_valid, y_pred)))
'''