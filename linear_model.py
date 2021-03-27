import numpy as np
from circArray import CircularArray
from sklearn import linear_model
import pickle

class LinModel():
    def __init__(self, buffer_size, X_shape, y_shape, min_samples=1, importance_sampling=True):
        self.buffer_X = CircularArray((buffer_size, *X_shape))
        self.buffer_y = CircularArray((buffer_size, *y_shape))
        self.model = linear_model.LinearRegression()
        self.fitted = False
        self.importance_sampling = importance_sampling
    
    def fit(self, X, y):
        for i in range(X.shape[0]):
            self.buffer_X.add(X[i])
            self.buffer_y.add(y[i])

        data_X = self.buffer_X.as_array()
        data_y = self.buffer_y.as_array().copy()

        if self.importance_sampling:
            if self.fitted:
                data_pred = self.predict(data_X)
                zeros = np.where(data_y==0)
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