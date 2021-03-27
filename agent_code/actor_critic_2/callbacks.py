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
from functions import augment, select_features, masked_mean_squared_error, to_features, handmade_features
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
    self.exp_buffer = np.zeros((10, 81))
    if self.train == False:
        self.model.actor = load_model('actor')
        #self.policy = TreeModel(10000, (56, ), (6, ), 10)
        #self.policy.load('policy.p')
        #self.policy = load_model('policy')
        #self.inverse = load_model('inverse')
        #self.encoder = load_model('encoder')
        #self.forward = load_model('forward')

def act(agent, game_state: dict):
    if game_state['step'] == 1:
        agent.exp_buffer = np.zeros((10, 81))
        agent.dist_actions = []

    features = handmade_features(game_state)
    agent.exp_buffer = np.roll(agent.exp_buffer, -1, axis=0)
    agent.exp_buffer[-1] = features

    dist_action = agent.model.actor.predict([agent.exp_buffer.reshape([1, 10, 81])])[0]

    action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=softmax(dist_action))

    agent.logger.info('Pick %s'%action)
    return action
