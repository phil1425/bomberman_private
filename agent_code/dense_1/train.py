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
from functions import augment, select_features, masked_mean_squared_error, to_features, to_labels, masked_mae, handmade_features, augment_game_states, augment_labels, rotate_game_states
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
        reward -= 0.1
    else:
        reward += 0.1

    if e.CRATE_DESTROYED in events and not e.KILLED_SELF in events:
        c = events.count(e.CRATE_DESTROYED)
        reward += c*0.5
    if e.COIN_COLLECTED in events:
        reward += 0.5
    if e.KILLED_OPPONENT in events:
        reward += 0.5
    if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
        reward += 0.5
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        reward -= 0.5
    if e.SURVIVED_ROUND in events:
        reward += 0.5
    
    return reward, inv_moves

def get_advantages(values, rewards, mask):
    gamma = 0.80
    lmbda = 0.85
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards)-1:
            delta = rewards[i] + gamma * values[i] - values[i]
        else:
            delta = rewards[i] + gamma * mask[i] * (values[i + 1] - values[i])

        gae = delta + gamma * lmbda * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values#[:-1]
    return np.array(returns), (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

def get_advantages_2(rewards, mask):
    lmbda = 0.8
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i]
        gae = delta + lmbda * gae * mask[i]
        returns.insert(0, gae)
    return np.array(returns)

def get_e_advantages(event_list, state_list, mask):
    lmbda = 0.5
    returns = []
    gae = 0
    for i in reversed(range(len(event_list))):
        reward_immedeate = 0
        reward = 0
        events = event_list[i]
        state = state_list[i]

        if e.INVALID_ACTION in events or e.WAITED in events:
            reward_immedeate -= 1
        else:
            reward_immedeate += 1

        if e.CRATE_DESTROYED in events and not e.KILLED_SELF in events:
            c = events.count(e.CRATE_DESTROYED)
            reward += c*0.1
        if e.COIN_COLLECTED in events:
            reward += 0.1
        if e.KILLED_OPPONENT in events:
            reward += 0.1
        if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
            reward += 0.1
        if e.KILLED_SELF in events or e.GOT_KILLED in events:
            reward -= 0.1
        if e.SURVIVED_ROUND in events:
            reward += 0.1

        delta = reward
        gae = delta + lmbda * gae * mask[i]
        returns.insert(0, gae+reward_immedeate)

    return returns

def build_dense():
    model = Sequential()
    model.add(Input(shape=(57,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(6, activation='linear'))
    model.compile(optimizer=Adam(3e-4), loss=mean_squared_error)
    return model

def setup_training(self):
    if False:
        pass

    else:
        #self.actor = LinModel(10000, (57,), (6,), 1, importance_sampling=True)
        self.actor = build_dense()
        self.critic = LinModel(10000, (57,), (1,), 1, importance_sampling=False)

    self.X = []
    self.y = []
    self.rewards = []
    self.mask = []
    self.events = []
    self.states = []

    self.num_invalid_moves = 0
    self.avg_reward = np.zeros(10)
    self.avg_invalid_actions = np.zeros(10)

def game_events_occurred(self, old_game_state, action, new_game_state, events):
    self.logger.info(events)
    if old_game_state != None:
        reward, inv_moves = calc_reward(events, old_game_state['round'])
        self.num_invalid_moves += inv_moves
        self.rewards.append(reward)
        self.events.append(events)
        self.states.append(old_game_state)

        state, rot_action, _, _ = rotate_game_states([old_game_state], [to_labels(action)])
        self.X.append(handmade_features(state[0]))
        self.y.append(rot_action[0])
        self.mask.append(1)

def end_of_round(self, last_game_state, last_action, events):
    self.logger.info(events)
    reward, inv_moves = calc_reward(events, last_game_state['round'])
    self.num_invalid_moves += inv_moves
    self.rewards.append(reward)
    self.events.append(events)
    self.states.append(last_game_state)

    state, action, _, _ = rotate_game_states([last_game_state], [to_labels(last_action)])
    self.X.append(handmade_features(state[0]))
    self.y.append(action[0])
    self.mask.append(0)

    X = np.array(self.X)
    y = np.array(self.y)
    rewards = np.array(self.rewards)
    mask = np.array(self.mask)

    #try:
    #    values = self.critic.predict(X).reshape((-1))
    #    prob_actions = self.actor.predict(X)
    #except sklearn.exceptions.NotFittedError:
    #    values = np.zeros(X.shape[0])
    #    prob_actions = y
    
    advantages = np.array(get_e_advantages(self.events, self.states, self.mask))
    
    #y_actor = y*advantages.reshape((-1, 1))
    #y_critic = returns.reshape((-1, 1))

    self.avg_invalid_actions = np.roll(self.avg_invalid_actions, -1)
    self.avg_invalid_actions[-1] = self.num_invalid_moves/len(self.rewards)
    self.avg_reward = np.roll(self.avg_reward, -1)
    self.avg_reward[-1] = np.mean(self.rewards)

    self.logger.info('invalid moves %f'%(self.num_invalid_moves/len(self.rewards)))
    self.logger.info('average invalid moves %f'%(np.mean(self.avg_invalid_actions)))
    self.logger.info('reward: %f'%(np.mean(reward)))
    self.logger.info('average reward: %f'%(np.mean(self.avg_reward)))

    if last_game_state['round']%2 == 0:
        prob_action = self.actor.predict(X)
        y_actor = prob_action
        y_actor[y==1] = advantages
        self.actor.fit(X, y_actor, batch_size=16, epochs=2)
        #self.critic.fit(X, y_critic)
        self.X = []
        self.y = []
        self.rewards = []
        self.values = []
        self.mask = []
        self.num_invalid_moves = 0
        self.events = []
        self.states = []
    else:
        pass
        #self.actor.add_data(X, y_actor)
        #self.critic.add_data(X, y_critic)

    if last_game_state['round'] % 100 == 0: 
        self.actor.save('actor')
        self.critic.save('critic')
