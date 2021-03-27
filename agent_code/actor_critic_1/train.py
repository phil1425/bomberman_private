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

def build_actor_2():
    state_input = Input((10, 81))
    oldpolicy_probs = Input(shape=(1, 6,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    x = LSTM(16, return_sequences=True)(state_input)
    x = LSTM(16)(x)
    out_actions = Dense(6, activation='softmax')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])

    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    return model

def build_actor():
    state_input = Input((10, 81))

    x = LSTM(32, return_sequences=True)(state_input)
    x = LSTM(32)(x)
    out_actions = Dense(6, activation='softmax')(x)

    model = Model(inputs=[state_input],
                  outputs=[out_actions])

    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss_2])
    return model

def build_critic():
    state_input = Input((10, 81))

    x = LSTM(16, return_sequences=True)(state_input)
    x = LSTM(16, return_sequences=False)(x)
    out_actions = Dense(1, activation='linear')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.summary()
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    return model

def calc_reward(events):
    reward = 0
    inv_moves = 0
    if e.INVALID_ACTION in events or e.WAITED in events:
        inv_moves += 1
        reward -= 1
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

def get_advantages(values, rewards):
    gamma = 0.5
    lmbda = 0.5
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards)-1:
            delta = rewards[i] + gamma * values[i] - values[i]
        else:
            delta = rewards[i] + gamma * values[i + 1] - values[i]

        gae = delta + gamma * lmbda * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values#[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    oldpolicy_probs = tf.convert_to_tensor(oldpolicy_probs, dtype=tf.float32)
    advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    values = tf.convert_to_tensor(values, dtype=tf.float32)

    clipping_val = 0.2
    critic_discount = 0.99
    entropy_beta = 0.01
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * K.reshape(advantages, (-1, 1))
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * K.reshape(advantages, (-1, 1))
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss

def ppo_loss_2(y_true, y_pred):
    advantages = K.reshape(y_true[:,0], (-1,1))
    rewards = K.reshape(y_true[:,1], (-1,1))
    values = K.reshape(y_true[:,2], (-1,1))
    oldpolicy_probs = y_true[:,3:]

    clipping_val = 0.2
    critic_discount = 0.99
    entropy_beta = 0.01

    newpolicy_probs = y_pred
    ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
    p1 = ratio * advantages
    p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
    actor_loss = -K.mean(K.minimum(p1, p2))
    critic_loss = K.mean(K.square(rewards - values))
    total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
        -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
    
    return total_loss

def setup_training(self):
    if False:
        pass

    else:
        self.actor = build_actor()
        self.critic = build_critic()

    self.aug_buffer_x = np.zeros((8, 10, 81))

    self.X = []
    self.y = []
    self.rewards = []
    self.values = []

    self.num_invalid_moves = 0

    self.avg_reward = np.zeros(10)
    self.avg_invalid_actions = np.zeros(10)

def game_events_occurred(self, old_game_state, action, new_game_state, events):
    self.logger.info(events)
    if old_game_state != None:
        add_experience(self, old_game_state, action)
        reward, inv_moves = calc_reward(events)
        self.num_invalid_moves+= inv_moves
        self.rewards.append(reward)

def end_of_round(self, last_game_state, last_action, events):
    self.logger.info(events)
    #self.logger.info(last_game_state['step'])
    add_experience(self, last_game_state, last_action)
    reward, inv_moves = calc_reward(events)
    self.num_invalid_moves += inv_moves
    self.rewards.append(reward)

    X = np.array(self.X)
    y = np.array(self.y)
    rewards = np.array(self.rewards)

    values = np.zeros((X.shape[0], 8))
    returns = np.zeros((X.shape[0], 8))
    advantages = np.zeros((X.shape[0], 8))

    for i in range(8):
        values[:,i] = self.critic.predict(X[:,i]).reshape((-1,))
        returns[:,i], advantages[:,i] = get_advantages(values[:,i], rewards)

    X_actor = X[:,0]
    values_actor = values[:,0]
    advantages_actor = advantages[:,0]
    rewards_actor = rewards.copy()
    dist_actions = self.actor.predict(X[:,0])

    values = values.reshape((-1,))
    advantages = advantages.reshape((-1,))
    returns = returns.reshape((-1,))
    X = X.reshape((-1, 10, 81))
    y = y.reshape((-1, 6))


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

    y_concat = np.stack([advantages_actor, rewards_actor, values_actor], axis=1)
    y_concat = np.concatenate([y_concat, dist_actions], axis=1)

    #self.actor.compile(Adam(1e-4), loss=[ppo_loss(dist_actions, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values)])
    self.actor.fit([X_actor], [y_concat], epochs=8, batch_size=4)
    self.critic.fit([X], [np.reshape(returns, newshape=(-1, 1))], epochs=8, batch_size=16)

    self.avg_invalid_actions = np.roll(self.avg_invalid_actions, -1)
    self.avg_invalid_actions[-1] = self.num_invalid_moves/len(self.rewards)
    self.avg_reward = np.roll(self.avg_reward, -1)
    self.avg_reward[-1] = np.mean(self.rewards)

    self.logger.info('invalid moves %f'%(self.num_invalid_moves/len(self.rewards)))
    self.logger.info('average invalid moves %f'%(np.mean(self.avg_invalid_actions)))
    self.logger.info('reward: %f'%(np.mean(reward)))
    self.logger.info('average reward: %f'%(np.mean(self.avg_reward)))

    self.X = []
    self.y = []
    self.rewards = []
    self.values = []
    self.aug_buffer_x = np.zeros((8, 10, 81))

    self.num_invalid_moves = 0
    if last_game_state['round'] % 100 == 0: 
        self.actor.save('actor')
        self.critic.save('critic')
