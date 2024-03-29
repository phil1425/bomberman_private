import numpy as np
from circArray import CircularArray
from sklearn import tree
import pickle

class TreeModel():
    def __init__(self, buffer_size, X_shape, y_shape, min_samples=1, max_leaf_nodes=None):
        self.buffer_X = CircularArray((buffer_size, *X_shape))
        self.buffer_y = CircularArray((buffer_size, *y_shape))
        self.model = tree.DecisionTreeRegressor(min_samples_leaf=min_samples, max_leaf_nodes=None)
        self.fitted = False
    
    def fit(self, X, y, lr=1):
        for i in range(X.shape[0]):
            self.buffer_X.add(X[i])
            self.buffer_y.add(y[i])

        data_X = self.buffer_X.as_array()
        data_y = self.buffer_y.as_array().copy()

        if self.fitted:
            data_pred = self.predict(data_X)
            zeros = np.where(data_y==0)
            data_y[data_y!=0] -= data_pred[data_y!=0]
            data_y[data_y!=0] *= lr
            data_y[data_y!=0] += data_pred[data_y!=0]
            data_y[zeros] = data_pred[zeros]

        else:
            data_pred = np.mean(data_y, axis=0)
            zeros = np.where(data_y==0)
            data_y[zeros] = data_pred[zeros[1]]

        self.model.fit(data_X, data_y)
        self.fitted = True

    def add_data(self, X, y):
        for i in range(X.shape[0]):
            self.buffer_X.add(X[i])
            self.buffer_y.add(y[i])
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        pickle.dump(self.model, open(path, "wb"))

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))