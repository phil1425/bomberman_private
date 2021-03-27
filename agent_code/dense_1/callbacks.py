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
from functions import augment, select_features, masked_mean_squared_error, to_features, handmade_features, rotate_game_states, reverse_rotate_action
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
    if agent.train:
        if game_state['round'] < 0:
            epsilon = 1
        else:
            epsilon = 0.05+0.1*np.exp(game_state['round']/500)
    else:
        epsilon = 0.05

    if np.random.random() > epsilon:
        state, action, rot, mir = rotate_game_states([game_state], [np.array([0,0,0,0,0,0])])
        features = handmade_features(state[0])
        dist_action = agent.actor.predict(features.reshape((1, 57)))
        rot_action = reverse_rotate_action(dist_action, rot, mir)[0]
        action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'][np.argmax(rot_action)]
        #action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=softmax(rot_action))
    else:
        p = np.array([1,1,1,1,0.1,0])
        p /= np.sum(p)
        action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=p)
    agent.logger.info('Pick %s'%action)
    return action
