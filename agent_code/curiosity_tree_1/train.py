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
from functions import augment, select_features, masked_mean_squared_error, to_features, to_labels, masked_mae, handmade_features, augment_game_states, augment_labels, rotate_game_states, minimal_features_2
from tree_model import TreeModel
from linear_model import LinModel
from features import FeatureSelector
import sklearn

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

encoding_dim = 16
def build_encoder():
    model = Sequential()
    model.add(Input(shape=(17, 17, 8)))
    model.add(Conv2D(8, 3, activation='relu', padding='same'))
    model.add(Conv2D(8, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(Conv2D(4, 1, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(encoding_dim, activation='linear'))

    model.summary()
    return model

def build_inverse(encoder):
    old_state = Input(shape=(17,17,12))
    new_state = Input(shape=(17,17,12))

    old_encoding = encoder(old_state)
    new_encoding = encoder(new_state)

    x = Concatenate(axis=-1)([old_encoding, new_encoding])
    x = Dense(64, activation='relu')(x)
    action = Dense(6, activation='softmax')(x)

    model = Model(inputs=[old_state, new_state], outputs=[action])

    model.summary()
    return model

def build_forward():
    model = Sequential()
    model.add(Input(shape=(encoding_dim+6)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(encoding_dim, activation='relu'))

    model.summary()
    return model

def calc_reward(events, n):
    i_reward = 0
    reward = 0
    if e.INVALID_ACTION in events or e.WAITED in events:
        i_reward -= 0.1+(1-np.exp(-n/500))
        reward -= 0.05
    else:
        i_reward += 0.1*np.exp(-n/500)
        reward -= 0.05

    if e.CRATE_DESTROYED in events and not e.KILLED_SELF in events:
        c = events.count(e.CRATE_DESTROYED)
        reward += c*0.1
    if e.COIN_COLLECTED in events:
        reward += 1
    if e.KILLED_OPPONENT in events:
        reward += 0.5
    if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
        reward += 0.05
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        reward -= 1
    if e.SURVIVED_ROUND in events:
        reward += 1

    return reward, i_reward

def get_returns(events, n):
    lmbda = 0.95
    returns = []
    gae = 0
    for i in reversed(range(len(events))):
        delta, i_reward = calc_reward(events[i], n)
        gae = delta + lmbda * gae
        returns.insert(0, gae+i_reward)
    return np.array(returns)

def setup_training(self):
    self.actor = TreeModel(10000, (32,), (6,), 10)
    self.feature_selector = FeatureSelector()

    self.features = []
    self.labels = []
    self.events = []
    # 0 steps, 1 inv_moves, 2 waited, 3 crates, 4 coins, 5 kills, 6 deaths, 7 suicides
    self.diagnostics = [0, 0, 0, 0, 0, 0, 0, 0]
    self.history = []
    self.avg_invalid_actions = np.zeros(10)

def update_diagnostics(self, events):
    self.diagnostics[0] += 1 
    self.diagnostics[1] += events.count(e.INVALID_ACTION)
    self.diagnostics[2] += events.count(e.WAITED)
    self.diagnostics[3] += events.count(e.CRATE_DESTROYED)
    self.diagnostics[4] += events.count(e.COIN_COLLECTED)
    self.diagnostics[5] += events.count(e.KILLED_OPPONENT)
    self.diagnostics[6] += events.count(e.GOT_KILLED)
    self.diagnostics[7] += events.count(e.KILLED_SELF)

def update_data(self, old_game_state, action, events, fit=False):
    state, rot_action, _, _ = rotate_game_states([old_game_state], [to_labels(action)])
    features = self.feature_selector.eval_all_features(state[0])
    self.features.append(features)
    self.labels.append(rot_action[0])
    self.events.append(events)

def fit_model(self):
    returns = get_returns(self.events, self.n)
    X_actor = np.array(self.features)
    y_actor = np.array(self.labels)*np.array(returns).reshape(-1, 1)
    self.actor.fit(X_actor, y_actor)

def game_events_occurred(self, old_game_state, action, new_game_state, events):
    self.logger.info(events)
    if old_game_state != None:
        update_data(self, old_game_state, action, events)
        update_diagnostics(self, events)

def end_of_round(self, last_game_state, last_action, events):
    self.logger.info(events)
    self.n = last_game_state['round']

    update_data(self, last_game_state, last_action, events, fit=True)
    update_diagnostics(self, events)
    fit_model(self)

    self.avg_invalid_actions = np.roll(self.avg_invalid_actions, -1)
    self.avg_invalid_actions[-1] = self.diagnostics[1]/self.diagnostics[0]
    self.logger.info('invalid moves %f'%(self.diagnostics[1]/self.diagnostics[0]))
    self.logger.info('average invalid moves %f'%(np.mean(self.avg_invalid_actions)))

    self.features = []
    self.labels = []
    self.events = []

    self.history.append(self.diagnostics)
    self.diagnostics = [0, 0, 0, 0, 0, 0, 0, 0]

    self.steps = 0

    if self.n%100 == 0:
        self.actor.save('actor.p')
        np.savetxt('diagnostics.csv', np.array(self.history))
