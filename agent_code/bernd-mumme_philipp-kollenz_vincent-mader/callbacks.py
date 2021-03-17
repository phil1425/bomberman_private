import h5py
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import mean_squared_error
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

def masked_mean_squared_error(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, 0))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, 0))
    return mean_squared_error(y_true_masked, y_pred_masked)

def handmade_features(game_state):
    data = []#np.zeros(64)
    coords = game_state['self'][3]
    x = coords[0]
    y = coords[1]
    data.append(game_state['step'])
    data.append(game_state['self'][1])

    #walls
    walls = (game_state['field']==-1).astype(int)
    walls[0,:] += 1
    walls[-1,:] += 1
    walls[:,0] += 1
    walls[:,-1] += 1
    data.append(walls[x, y+1])
    data.append(walls[x, y-1])
    data.append(walls[x+1, y])
    data.append(walls[x-1, y])
    #agent.logger.info('walls:'+str(data[-4:]))

    #crates
    crates = (game_state['field']==1).astype(int)
    crates_1 = np.pad(crates, 2)
    crates_1 = crates_1[x:x+5, y:y+5].reshape(25)
    mask = np.full(25, True)
    mask[13] = False
    data.extend(crates_1[mask])
    #agent.logger.info('crates1:'+str(crates_1))

    crates_2 = np.pad(crates, 2)
    crates_2[x:x+5, y:y+5] = np.zeros((5,5))
    crates_3 = np.zeros(8)
    crates_3[0] = np.sum(crates_2[x, :y])
    crates_3[1] = np.sum(crates_2[x, y+1:])
    crates_3[2] = np.sum(crates_2[x+1:, y])
    crates_3[3] = np.sum(crates_2[:x, y])

    crates_3[4] = np.sum(crates_2[x+1:, y+1:])
    crates_3[5] = np.sum(crates_2[:x, y+1:])
    crates_3[6] = np.sum(crates_2[x+1:, :y])
    crates_3[7] = np.sum(crates_2[:x, :y])
    data.extend(crates_3)
    #agent.logger.info('crates3:'+str(crates_3))

    bomb_map = np.zeros((17+6, 17+6))
    bomb_cooldowns = [x[1] for x in game_state['bombs']]
    bomb_idx_sorted = np.flip(np.argsort(bomb_cooldowns))

    for idx in bomb_idx_sorted:
        bomb = game_state['bombs'][idx]
        b_x = bomb[0][0]
        b_y = bomb[0][1]
        bomb_map[b_x:b_x+7, b_y+3] = bomb[1]
        bomb_map[b_x+3, b_y:b_y+7] = bomb[1]
    #agent.logger.info('bombs__map:'+str(bomb_map))

    bombs_x = bomb_map[x+3, y:y+7]
    bombs_y = bomb_map[x:x+7, y+3]
    #agent.logger.info('bombs_x:'+str(bombs_x))
    #agent.logger.info('bombs_y:'+str(bombs_y))

    data.extend(bombs_x)
    data.extend(bombs_y)

    c_coin = [0, 0]
    data_dist_coin = []
    for coin in game_state['coins']:
        dist = np.sqrt(np.sum(np.square([coin[0]-x, coin[1]-y])))
        data_dist_coin.append(dist)
    if data_dist_coin != []:
        c_coin[0] = x-game_state['coins'][np.argmin(data_dist_coin)][0]
        c_coin[1] = y-game_state['coins'][np.argmin(data_dist_coin)][1]

    data.extend(c_coin)
    #agent.logger.info('coin:'+str(c_coin))

    c_agent = [0, 0]
    data_dist_agent = []
    for agent in game_state['others']:
        dist = np.sqrt(np.sum(np.square([agent[3][0]-x, agent[3][1]-y])))
        data_dist_agent.append(dist)
    if data_dist_agent != []:
        c_agent[0] = x-game_state['others'][np.argmin(data_dist_agent)][3][0]
        c_agent[1] = y-game_state['others'][np.argmin(data_dist_agent)][3][1]

    data.extend(c_agent)
    #agent.logger.info('agent:'+str(c_agent))

    return np.array(data)

def setup(self):
    self.exp_buffer = np.zeros((10, 56))
    if self.train == False:
        self.policy = load_model('policy', custom_objects={'masked_mean_squared_error':masked_mean_squared_error})

def act(agent, game_state: dict):
    if game_state['step'] == 1:
        agent.exp_buffer = np.zeros((10, 56))

    if agent.train:
        if game_state['round'] == 1:
            epsilon = 1
        else:
            epsilon = 0.1#+0.3*np.exp(-game_state['round']/100)
    else:
        epsilon = 0.05

    features = handmade_features(game_state)
    agent.exp_buffer = np.roll(agent.exp_buffer, -1, axis=0)
    agent.exp_buffer[-1] = features

    if np.random.random()>epsilon:
        #agent.logger.info(agent.exp_buffer)
        dist_action = agent.policy.predict(agent.exp_buffer.reshape([1, 10, 56]))[0]
        #dist_action = np.clip(dist_action, 0, 99)+1e-5
        #dist_action /= np.sum(dist_action)
        action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'][np.argmax(dist_action)]
        #action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'], p=dist_action)
    else:
        dist_action = np.array([1., 1., 1., 1., 1])
        dist_action /= np.sum(dist_action)
        action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=dist_action)

    agent.logger.info('Pick %s'%action)
    return action
