import numpy as np
import pickle
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, Input, Concatenate, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, mse, mae
from tensorflow.keras.metrics import mean_absolute_error
import events as e
import matplotlib.pyplot as plt
import tensorflow as tf
from functions import augment, select_features, masked_mean_squared_error, to_features, to_labels, masked_mae, handmade_features
from tree_model import TreeModel

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def setup_training(self):
    if False:
        #self.policy = load_model('policy', custom_objects={'masked_mean_squared_error':masked_mean_squared_error})
        self.policy = TreeModel(4000, (64, ), (6, ))
        self.policy.load('policy.p')

    else:
        self.policy = TreeModel(4000, (56, ), (6, ), 1)

    self.new_features = []
    self.old_features = []
    self.labels = []
    self.aux_reward = []

    self.num_invalid_moves = 0

def add_to_list(data, value, steps):
    if len(data) > steps:
        for i in range(0, steps):
            data[-i-1] += value
    else:
        for i in range(len(data)):
            data[i] += value
    
    return data

def game_events_occurred(self, old_game_state, action, new_game_state, events):
    self.logger.info(events)

    if old_game_state != None and new_game_state != None:

        self.aux_reward.append(0)
        if e.INVALID_ACTION in events or e.WAITED in events:
            self.num_invalid_moves += 1
            self.aux_reward = add_to_list(self.aux_reward, -1, 1)
        else:
            self.aux_reward = add_to_list(self.aux_reward, 1, 1)

        if e.CRATE_DESTROYED in events:
            self.aux_reward = add_to_list(self.aux_reward, 0.5, 5)
        if e.COIN_FOUND in events:
            self.aux_reward = add_to_list(self.aux_reward, 0.5, 5)
        if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
            self.aux_reward = add_to_list(self.aux_reward, 1, 6)
        if e.KILLED_SELF in events:
            self.aux_reward = add_to_list(self.aux_reward, -5, 6)

        self.old_features.append(handmade_features(old_game_state))
        self.new_features.append(handmade_features(new_game_state))

        self.labels.append(to_labels(action))

def end_of_round(self, last_game_state, last_action, events):
    self.logger.info(events)
    self.aux_reward.append(0)
    if e.KILLED_SELF in events:
        self.aux_reward = add_to_list(self.aux_reward, -5, 6)

    self.new_features.append(to_features(last_game_state))
    self.old_features.append(self.new_features[-2])
    self.labels.append(to_labels(last_action))

    old_features = np.array(self.old_features)
    aux_reward = np.array(self.aux_reward)
    labels = np.array(self.labels)

    self.logger.info('invalid moves %f'%(self.num_invalid_moves/len(self.labels)))

    reward = aux_reward#+(last_game_state['self'][1]/last_game_state['step'])
    y_policy = labels*reward[:,None]
    X_policy = old_features

    self.policy.fit(X_policy, y_policy)

    self.new_features = []
    self.old_features = []
    self.labels = []
    self.aux_reward = []

    self.num_invalid_moves = 0
    if last_game_state['round'] % 100 == 0: 
        self.policy.save('policy.p')
