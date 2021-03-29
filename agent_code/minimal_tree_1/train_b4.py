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

def calc_reward(self, events, n, i):
    i_reward = 0
    reward = 0
    if e.INVALID_ACTION in events or e.WAITED in events:
        i_reward -= 1
        reward -= 0.05
    else:
        i_reward += 0.2
        reward -= 0.05

    if e.CRATE_DESTROYED in events and not e.KILLED_SELF in events:
        c = events.count(e.CRATE_DESTROYED)
        reward += c
    if e.COIN_COLLECTED in events:
        reward += 1
    if e.KILLED_OPPONENT in events:
        reward += 5#*(1-np.exp(-n/1000))
    if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
        reward += 0.5
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        reward -= 5#+4*(1-np.exp(-n/1000))
    if e.SURVIVED_ROUND in events:
        reward += 2#+4*(1-np.exp(-n/500))

    #movements = self.movements[:i]

    #if len(movements) > 4:
    #    if movements[-2:] == movements[-4:-2]:
    #        i_reward -= 0.05

    #if len(movements) > 6:
    #    if movements[-3:] == movements[-6:-3]:
    #        i_reward -= 0.05

    position = self.visited[i]
    self.logger.info(position)
    self.logger.info(self.visited)
    if i > 1:
        if position in self.visited[:i-1]:
            i_reward -= 0.1
        else:
            i_reward += 0.1

    return reward, i_reward

def get_returns(self, events, n):
    lmbda = 0.90
    returns = []
    gae = 0
    for i in reversed(range(len(events))):
        delta, i_reward = calc_reward(self, events[i], n, i)
        gae = delta + lmbda * gae
        returns.insert(0, gae+i_reward)
    return np.array(returns)

def setup_training(self):
    if False:
        self.actor = TreeModel(1000000, (24,), (6,), 20)
    else:
        self.actor = TreeModel(1000000, (24,), (6,), 20)
        self.actor.load('actor.p')
        data_X = np.load('actor_X.npy')
        data_y = np.load('actor_y.npy')
        self.actor.add_data(data_X, data_y)

    self.feature_selector = FeatureSelector()

    self.features = []
    self.labels = []
    self.events = []
    self.movements = []
    self.visited = []

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

    if e.MOVED_UP in events:
        mov = 0
    elif e.MOVED_DOWN in events:
        mov = 1
    elif e.MOVED_LEFT in events:
        mov = 2
    elif e.MOVED_RIGHT in events:
        mov = 3
    else:
        mov = 4
    self.movements.append(mov)
    self.visited.append(old_game_state['self'][3])

def fit_model(self):
    X = np.array(self.features)
    returns = get_returns(self, self.events, self.n)

    y_actor = np.array(self.labels)*np.array(returns).reshape(-1, 1)
    if self.n % 20 == 0 or self.n == 1:
        self.actor.fit(X, y_actor)
    else:
        self.actor.add_data(X, y_actor)

    #y_critic = np.array(returns).reshape(-1, 1)
    #self.critic.fit(X, y_critic)

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

    self.movements = []
    self.visited = []

    self.history.append(self.diagnostics)
    self.diagnostics = [0, 0, 0, 0, 0, 0, 0, 0]

    self.steps = 0

    if self.n%100 == 0:
        self.actor.save('actor.p')
        np.save("actor_X.npy", self.actor.buffer_X.as_array())
        np.save("actor_y.npy", self.actor.buffer_y.as_array())
        np.savetxt('diagnostics.csv', np.array(self.history))
