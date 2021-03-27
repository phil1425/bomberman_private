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
from functions import augment, select_features, masked_mean_squared_error, to_features, to_labels, masked_mae, handmade_features, augment_game_states
from tree_model import TreeModel

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def build_lstm():
    model = Sequential()
    model.add(Input(shape=(10, )))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(6, activation='linear'))

    model.summary()
    return model

def setup_training(self):
    if False:
        self.policy = load_model('policy')
        self.policy.compile(Adam(3e-4), loss=masked_mean_squared_error)
        #self.policy = TreeModel(4000, (64, ), (6, ))
        #self.policy.load('policy.p')

    else:
        self.policy = build_lstm()
        self.policy.compile(Adam(3e-4), loss=mean_squared_error)

    self.aux_reward = []
    self.aug_buffer_x = np.zeros((8, 10, 81))
    self.aug_buffer_reward = np.zeros((8, 10))

    self.X = []
    self.y = []

    self.num_invalid_moves = 0
    self.num_boxes_destroyed = 0

    self.avg_reward = np.zeros(10)
    self.avg_invalid_actions = np.zeros(10)

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
    n = new_game_state['round']
    a_factor = np.exp(-n/2000)
    b_factor = 1-np.exp(-n/2000)
    #self.logger.info(self.exp_buffer)

    if old_game_state != None and new_game_state != None:

        self.aux_reward.append(0)
        if e.INVALID_ACTION in events or e.WAITED in events:
            self.num_invalid_moves += 1
            self.aux_reward = add_to_list(self.aux_reward, -1-4*b_factor, 1)
        else:
            self.aux_reward = add_to_list(self.aux_reward, 1*a_factor, 1)

        if e.CRATE_DESTROYED in events and not e.KILLED_SELF in events:
            c = events.count(e.CRATE_DESTROYED)
            self.num_boxes_destroyed += c
            self.aux_reward = add_to_list(self.aux_reward, 1*c, 10)
        if e.COIN_COLLECTED in events:
            self.aux_reward = add_to_list(self.aux_reward, 5, 5)
        if e.KILLED_OPPONENT in events:
            self.aux_reward = add_to_list(self.aux_reward, 5, 7)
        if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
            self.aux_reward = add_to_list(self.aux_reward, 1+4*b_factor, 5)
        if e.KILLED_SELF in events:
            self.aux_reward = add_to_list(self.aux_reward, -1-4*b_factor, 7)

        aug_states, aug_labels = augment_game_states([old_game_state], [to_labels(action)])
        aug_states = [handmade_features(s) for s in aug_states]
        self.aug_buffer_x = np.roll(self.aug_buffer_x, -1, axis=1)

        self.aug_buffer_x[:,-1,:] = np.array(aug_states)
        self.new_game_state = new_game_state

        self.y.append(aug_labels)
        self.X.append(self.aug_buffer_x.copy())

def end_of_round(self, last_game_state, last_action, events):
    n = last_game_state['round']
    b_factor = 1-np.exp(-n/2000)
    c_factor = np.exp(-n/5000)
    self.logger.info(events)
    self.aux_reward.append(0)
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        self.aux_reward = add_to_list(self.aux_reward, -1-4*b_factor, 7)

    aug_states, aug_labels = augment_game_states([self.new_game_state], [to_labels(last_action)])
    aug_states = [handmade_features(s) for s in aug_states]
    self.aug_buffer_x = np.roll(self.aug_buffer_x, -1, axis=1)

    self.aug_buffer_x[:,-1] = np.array(aug_states)
    self.X.append(self.aug_buffer_x.copy())
    self.y.append(aug_labels)

    aux_reward = np.array(self.aux_reward)
    X = np.array(self.X).reshape([-1, 10, 81])
    y = np.array(self.y).reshape([-1, 6])



    reward = c_factor*aux_reward+(1-c_factor)*(last_game_state['self'][1])
    reward = np.repeat(reward, 8)

    y_new = y*reward[:,None]
    X_policy = X
    y_pred = self.policy.predict(X_policy)
    y_policy = y_pred
    y_policy[y==1] = y_new[y==1]
    #self.logger.info(X)
    #self.logger.info(y)

    self.avg_invalid_actions = np.roll(self.avg_invalid_actions, -1)
    self.avg_invalid_actions[-1] = self.num_invalid_moves/len(self.aux_reward)

    self.avg_reward = np.roll(self.avg_reward, -1)
    self.avg_reward[-1] = np.mean(reward)

    self.logger.info('invalid moves %f'%(self.num_invalid_moves/len(self.aux_reward)))
    self.logger.info('average invalid moves %f'%(np.mean(self.avg_invalid_actions)))
    self.logger.info('reward: %f'%(np.mean(reward)))
    self.logger.info('average reward: %f'%(np.mean(self.avg_reward)))

    self.policy.fit(X_policy, y_policy, batch_size=4, epochs=1)
    
    self.labels = []
    self.aux_reward = []
    self.X = []
    self.y = []

    self.game_state = []
    self.action = []

    self.num_invalid_moves = 0
    self.num_boxes_destroyed = 0
    if last_game_state['round'] % 100 == 0: 
        self.policy.save('policy')
