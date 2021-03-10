import numpy as np
import h5py
import numpy as np
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

#def setup(self):
#    np.random.seed()
#
#def act(agent, game_state: dict):
#    agent.logger.info('Pick action at random')
#    action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
#    return action

def augment(features, labels):
    new_features = np.zeros([4, *features.shape])
    new_labels = np.zeros([4, *labels.shape])

    new_features[0] = features
    new_features[1] = np.flip(features, axis=1)
    new_features[2] = np.flip(features, axis=2)
    new_features[3] = np.flip(features, axis=(1,2))
    #new_features[4] = np.rot90(features, axes=(1,2))
    #new_features[5] = np.rot90(new_features[4], axes=(1,2))
    #new_features[6] = np.rot90(new_features[5], axes=(1,2))

    new_labels[0] = labels
    new_labels[1] = labels[:,[1, 0, 2, 3, 4, 5]]
    new_labels[2] = labels[:,[0, 1, 3, 2, 4, 5]]
    new_labels[3] = labels[:,[1, 0, 3, 2, 4, 5]]
    #new_labels[4] = labels[[0, 1, 2, 3, 4, 5]]
    #new_labels[5] = labels[[0, 1, 2, 3, 4, 5]]
    #new_labels[6] = labels[[0, 1, 2, 3, 4, 5]]

    return np.concatenate(new_features, axis=0), np.concatenate(new_labels, axis=0)

def masked_mean_squared_error(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, 0))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, 0))
    return mean_squared_error(y_true_masked, y_pred_masked)

def to_features(game_state):
    data = np.zeros((*game_state['field'].shape, 12))
    data[:,:,0] = (game_state['field']==1).astype(int)   
    data[:,:,1] = (game_state['field']==-1).astype(int)

    for bomb in game_state['bombs']:
        data[bomb[0][0], bomb[0][1], 2+bomb[1]] = 1

    data[:,:,6] = game_state['explosion_map']
    for coin in game_state['coins']:
        data[coin[0], coin[1], 7] = 1

    data[game_state['self'][3][0], game_state['self'][3][1], 8] = 1
    data[game_state['self'][3][0], game_state['self'][3][1], 9] = int(game_state['self'][2])

    for other in game_state['others']:
        data[other[3][0], other[3][1], 10] = 1
        data[other[3][0], other[3][1], 11] = int(other[2])
    
    return data

def select_features(X):
    X_new = np.zeros((7, 7, 12))
    data = np.pad(X, pad_width=((3,3), (3,3), (0,0)))
    coords = np.where(data[8]==1)[0]
    X_new = data[coords[0]-3:coords[0]+4, coords[1]-3:coords[1]+4]
    return X_new

def setup(self):
    self.model = load_model('model', custom_objects={'masked_mean_squared_error':masked_mean_squared_error})
    #features = np.load('features.npy')
    #labels = np.load('labels.npy')
    #selectiveModel()#LinearRegression()#tree.DecisionTreeRegressor()
    #self.model.fit(*augment(features, labels))
    pass

def act(agent, game_state: dict):
    r = game_state['round']

    features = select_features(to_features(game_state)).reshape([1, 7, 7, 12])
    dist_action = agent.model.predict(features)[0]
    dist_action -= np.mean(dist_action)
    dist_action = np.clip(dist_action, 0, 100)
    dist_action = np.nan_to_num(dist_action)+5e-10

    dist_action /= np.sum(dist_action)
    agent.logger.info(dist_action)

    #action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'WAIT', 'BOMB'], p=[.2, .2, .2, .2, .1, .1])
    action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=dist_action)
    #action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'][np.argmax(dist_action)]
    agent.logger.info('Pick %s'%action)
    return action

