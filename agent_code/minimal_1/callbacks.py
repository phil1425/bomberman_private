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
from functions import augment, select_features, masked_mean_squared_error, to_features, handmade_features, rotate_game_states, reverse_rotate_action, minimal_features
from tree_model import TreeModel
from scipy.special import softmax
from circArray import CircularArray

#def setup(self):
#    np.random.seed()
#
#def act(agent, game_state: dict):
#    agent.logger.info('Pick action at random')
#    action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
#    return action

def setup(self):
    if self.train == False:
        self.actor = TreeModel(2000, (57,), (6,), 10)
        self.actor.loda('actor.p')

def act(agent, game_state: dict):
    epsilon = 0.05

    if np.random.random() > epsilon:
        state, action, rot, mir = rotate_game_states([game_state], [np.array([0,0,0,0,0,0])])
        features =  minimal_features(state[0])

        #p = np.zeros(6)
        #if features[1] == 0:
        #    p[1] = 1
        #if features[2] == 0:
        #    p[0] = 1
        #if features[3] == 0:
        #    p[3] = 1
        #if features[4] == 0:
        #    p[2] = 0
        #if features[0] == 1:
        #    p[-1] = 0
        #
        #if np.sum(p) == 0:
        #    p[-1] = 1

        try:
            idx_fit = np.where((agent.qtable_x == features).all(axis=1))[0][0]
        except IndexError:
            idx_fit = None
    
        if idx_fit != None:
            p = agent.qtable_y[idx_fit]
            p = reverse_rotate_action(p, rot, mir)
            p = np.clip(p, 0, 1)+1e-5
            p /= np.sum(p)
            action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=p)
        else:
            p = np.array([1,1,1,1,0.1,0])
            p /= np.sum(p)
            action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=p)

    else: 
        p = np.array([1,1,1,1,0.1,0])
        p /= np.sum(p)
        action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=p)

    agent.logger.info('Pick %s'%action)
    return action
