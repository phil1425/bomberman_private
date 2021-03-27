import numpy as np
import pickle
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, Input, Concatenate, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy, mse, mae
from tensorflow.keras.metrics import mean_absolute_error
import keras.backend as K
import events as e
import matplotlib.pyplot as plt
import tensorflow as tf
from functions import augment, select_features, masked_mean_squared_error, to_features, to_labels, masked_mae, handmade_features, augment_game_states, augment_labels, rotate_game_states, minimal_features
from tree_model import TreeModel
from linear_model import LinModel
import sklearn

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def calc_reward(events, states):
    reward = 0
    inv_moves = 0
    if e.INVALID_ACTION in events or e.WAITED in events:
        inv_moves += 1
        reward -= 1
    else:
        reward += 1

    if e.CRATE_DESTROYED in events and not e.KILLED_SELF in events:
        c = events.count(e.CRATE_DESTROYED)
        reward += c*0
    if e.COIN_COLLECTED in events:
        reward += 0
    if e.KILLED_OPPONENT in events:
        reward += 0
    if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
        reward += 0
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        reward -= 0
    if e.SURVIVED_ROUND in events:
        reward += 0
    
    return reward, inv_moves

def setup_training(self):
    self.qtable_x = np.zeros((100, 5))
    self.qtable_y = np.zeros((100, 6))
    self.num_invalid_moves = 0
    self.steps = 0
    self.avg_invalid_actions = np.zeros(10)

def update_qtable(self, old_game_state, action, events):
    reward, inv_moves = calc_reward(events, old_game_state['round'])
    self.num_invalid_moves += inv_moves
    state, rot_action, _, _ = rotate_game_states([old_game_state], [to_labels(action)])
    features = minimal_features(state[0])

    try:
        idx_fit = np.where((self.qtable_x == features).all(axis=1))[0][0]
    except IndexError:
        idx_fit = None
    

    features = np.array(features)
    rot_action = np.array(rot_action)

    if idx_fit != None:
        self.qtable_y[idx_fit][np.argmax(rot_action)] = reward

    else:
        self.qtable_x = np.array(np.roll(self.qtable_x, -1, axis=0))
        self.qtable_y = np.array(np.roll(self.qtable_y, -1, axis=0))
        self.qtable_x[-1] = np.array(features)
        self.qtable_y[-1] = np.array(rot_action*reward)

def game_events_occurred(self, old_game_state, action, new_game_state, events):
    self.logger.info(events)
    if old_game_state != None:
        update_qtable(self, old_game_state, action, events)
        self.steps += 1

def end_of_round(self, last_game_state, last_action, events):
    self.logger.info(events)
    update_qtable(self, last_game_state, last_action, events)
    self.steps += 1

    self.avg_invalid_actions = np.roll(self.avg_invalid_actions, -1)
    self.avg_invalid_actions[-1] = self.num_invalid_moves/self.steps

    self.logger.info('invalid moves %f'%(self.num_invalid_moves/self.steps))
    self.logger.info('average invalid moves %f'%(np.mean(self.avg_invalid_actions)))
    self.steps = 0
    self.num_invalid_moves = 0
