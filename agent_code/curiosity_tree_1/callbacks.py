import numpy as np
import h5py
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import os
import matplotlib.pyplot as plt
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from circBuffer import CircularBuffer
from functions import augment, select_features, masked_mean_squared_error, to_features, handmade_features, rotate_game_states, reverse_rotate_action, minimal_features_2
from tree_model import TreeModel
from scipy.special import softmax
from circArray import CircularArray
from features import FeatureSelector

#def setup(self):
#    np.random.seed()
#
#def act(agent, game_state: dict):
#    agent.logger.info('Pick action at random')
#    action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
#    return action

def setup(self):
    if not self.train:
        self.feature_selector = FeatureSelector()
        self.actor = TreeModel(10000, (16,), (6,), 100)
        self.actor.load('actor.p')

def act(self, game_state: dict):
    if self.train:
        if game_state['round'] == 1:
            epsilon = 1
        else:
            epsilon = 0.05+0.5*np.exp(-game_state['round']/200)
    else:
        epsilon = 0.05

    if np.random.random() > epsilon:
        state, action, rot, mir = rotate_game_states([game_state], [np.array([0,0,0,0,0,0])])
        features =  self.feature_selector.eval_all_features(state[0])

        p = self.actor.predict(features.reshape((1, -1)))[0]
        #gent.logger.info(p)
        p = reverse_rotate_action(p, rot, mir)
        action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'][np.argmax(p)]
        #action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=softmax(p))

    else: 
        p = np.array([1,1,1,1,0.1,0])
        p /= np.sum(p)
        action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=p)

    self.logger.info('Pick %s'%action)
    return action
