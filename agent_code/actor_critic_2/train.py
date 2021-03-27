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
from functions import augment, select_features, masked_mean_squared_error, to_features, to_labels, masked_mae, handmade_features, augment_game_states, augment_labels
from tree_model import TreeModel

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class ActorCritic():
    def __init__(self):
        self.common = self.build_common()
        self.actor = self.build_actor(self.common)
        self.critic = self.build_critic(self.common)

    def build_common(self):
        model = Sequential()
        model.add(LSTM(16, return_sequences=True))

        return model

    def build_actor(self, common):
        state_input = Input(shape=(10, 81))
        x = common(state_input)
        x = LSTM(8)(x)
        action_output = Dense(6, activation='linear')(x)
        model = Model(inputs=[state_input], outputs=[action_output])
        model.compile(Adam(3e-4), loss='mse')
        model.summary()

        return model

    def build_critic(self, common):
        state_input = Input(shape=(10, 81))
        x = common(state_input)
        x = LSTM(8)(x)
        action_output = Dense(1, activation='linear')(x)
        model = Model(inputs=[state_input], outputs=[action_output])
        model.compile(Adam(3e-4), loss='mse')
        model.summary()

        return model

def calc_reward(events):
    reward = 0
    inv_moves = 0
    if e.INVALID_ACTION in events or e.WAITED in events:
        inv_moves += 1
        reward -= 0.1
    else:
        reward += 0.01

    if e.CRATE_DESTROYED in events and not e.KILLED_SELF in events:
        c = events.count(e.CRATE_DESTROYED)
        reward += c*0.05
    if e.COIN_COLLECTED in events:
        reward += 0.5
    if e.KILLED_OPPONENT in events:
        reward += 0.5
    if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
        reward += 0.1
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        reward -= 0.5
    if e.SURVIVED_ROUND in events:
        reward += 0.5 
    
    return reward, inv_moves

def add_experience(self, game_state, action):
    aug_states, aug_labels = augment_game_states([game_state], [to_labels(action)])
    aug_states = [handmade_features(s) for s in aug_states]
    self.aug_buffer_x = np.roll(self.aug_buffer_x, -1, axis=1)
    self.aug_buffer_x[:,-1,:] = np.array(aug_states)

    self.y.append(aug_labels)
    self.X.append(self.aug_buffer_x.copy())

def get_advantages(values, rewards, mask):
    gamma = 0.95
    lmbda = 0.99
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards)-1:
            delta = rewards[i]
        else:
            delta = rewards[i] + gamma * values[i + 1] - values[i]

        gae = delta + gamma * lmbda * gae * mask[i]
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values#[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

def setup_training(self):
    if False:
        pass

    else:
        self.model = ActorCritic()

    self.aug_buffer_x = np.zeros((8, 10, 81))

    self.X = []
    self.y = []
    self.rewards = []
    self.mask = []

    self.num_invalid_moves = 0
    self.num_steps = 0

    self.avg_reward = np.zeros(10)
    self.avg_invalid_actions = np.zeros(10)

def game_events_occurred(self, old_game_state, action, new_game_state, events):
    self.logger.info(events)
    if old_game_state != None:
        add_experience(self, old_game_state, action)
        reward, inv_moves = calc_reward(events)
        self.num_invalid_moves+= inv_moves
        self.rewards.append(reward)
        self.mask.append(0)
        self.num_steps += 1

def end_of_round(self, last_game_state, last_action, events):
    self.logger.info(events)
    #self.logger.info(last_game_state['step'])
    add_experience(self, last_game_state, last_action)
    reward, inv_moves = calc_reward(events)
    self.num_invalid_moves += inv_moves
    self.num_steps += 1
    self.rewards.append(reward)
    self.mask.append(0)

    if last_game_state['round']%8==0:

        X = np.array(self.X)
        y = np.array(self.y)
        rewards = np.array(self.rewards)

        values = np.zeros((X.shape[0], 8))
        returns = np.zeros((X.shape[0], 8))
        advantages = np.zeros((X.shape[0], 8))
        for i in range(8):
            values[:,i] = self.model.critic.predict(X[:,i]).reshape((-1,))
            returns[:,i], advantages[:,i] = get_advantages(values[:,i], rewards, self.mask)
        dist_actions = self.model.actor.predict(X.reshape(-1, 10, 81))

        values = values.reshape((-1,))
        advantages = advantages.reshape((-1,))
        returns = returns.reshape((-1,))
        X = X.reshape((-1, 10, 81))
        y = y.reshape((-1, 6))
        dist_actions = dist_actions.reshape((-1, 6))

        y_actor = dist_actions
        y_actor[y==1] = advantages

        y_critic = returns.reshape((-1,1))

        #self.logger.info(X.shape)
        #self.logger.info(y.shape)
        #self.logger.info(values.shape)
        #self.logger.info(advantages.shape)
        #self.logger.info(rewards.shape)
        #self.logger.info(returns)#.shape)
        #self.logger.info(dist_actions)#.shape)
        #self.logger.info(rewards_actor.shape)
        #self.logger.info(values_actor.shape)
        #self.logger.info(advantages_actor.shape)
        #self.logger.info(dist_actions.shape)

        rewards = np.repeat(rewards, 8)

        self.model.actor.fit(X, y_actor, epochs=4, batch_size=64)
        self.model.critic.fit(X, y_critic, epochs=4, batch_size=64)

        self.avg_invalid_actions = np.roll(self.avg_invalid_actions, -1)
        self.avg_invalid_actions[-1] = self.num_invalid_moves/self.num_steps
        self.avg_reward = np.roll(self.avg_reward, -1)
        self.avg_reward[-1] = np.mean(self.rewards)

        self.logger.info('invalid moves %f'%(self.num_invalid_moves/self.num_steps))
        self.logger.info('average invalid moves %f'%(np.mean(self.avg_invalid_actions)))
        self.logger.info('reward: %f'%(np.mean(rewards)))
        self.logger.info('average reward: %f'%(np.mean(self.avg_reward)))

        self.X = []
        self.y = []
        self.rewards = []
        self.mask = []
        self.num_invalid_moves = 0
        self.num_steps = 0

    self.aug_buffer_x = np.zeros((8, 10, 81))

    if last_game_state['round'] % 100 == 0: 
        self.model.common.save('common')
        self.model.actor.save('actor')
        self.model.critic.save('critic')
