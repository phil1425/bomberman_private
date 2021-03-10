import numpy as np
from circArray import CircularArray
from sklearn import tree
import pickle

class TreeModel():
    def __init__(self, buffer_size, X_shape, y_shape):
        self.buffer_X = CircularArray((buffer_size, *X_shape))
        self.buffer_y = CircularArray((buffer_size, *y_shape))
        self.model = tree.DecisionTreeRegressor()
    
    def fit(self, X, y):
        for i in range(X.shape[0]):
            self.buffer_X.add(X[i])
            self.buffer_y.add(y[i])

        data_X = self.buffer_X.buffer
        data_y = self.buffer_y.buffer.copy()
        
        data_pred = np.mean(data_y, axis=0)
        zeros = np.where(data_y==0)
        data_y[zeros] = data_pred[zeros[1]]

        self.model.fit(data_X, data_y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        pickle.dump(self.model, open(path, "wb"))

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))