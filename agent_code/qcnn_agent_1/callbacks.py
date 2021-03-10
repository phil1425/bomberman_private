import numpy as np
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

'''
def to_features(game_state):
    data = np.zeros((12,*game_state['field'].shape))
    data[0] = (game_state['field']==1).astype(int)   
    data[1] = (game_state['field']==-1).astype(int)

    for bomb in game_state['bombs']:
        data[2+bomb[1], bomb[0][0], bomb[0][1]] = 1

    data[6] = game_state['explosion_map']
    for coin in game_state['coins']:
        data[7, coin[0], coin[1]] = 1

    data[8, game_state['self'][3][0], game_state['self'][3][1]] = 1
    data[9, game_state['self'][3][0], game_state['self'][3][1]] = int(game_state['self'][2])

    for other in game_state['others']:
        data[10, other[3][0], other[3][1]] = 1
        data[11, other[3][0], other[3][1]] = int(other[2])
    
    return data
'''

def to_features(game_state):
    data = np.zeros((*game_state['field'].shape, 12))
    data[:,:,0] = (game_state['field']==1).astype(int)   
    data[:,:,1] = (game_state['field']==-1).astype(int)

    for bomb in game_state['bombs']:
        data[bomb[0][0], bomb[0][1], 2+bomb[1]] = 1

    data[:,:,6] = game_state['explosion_map']
    for coin in game_state['coins']:
        data[coin[0], coin[1], 7] = 1

    data[game_state['self'][3][0], game_state['self'][3][1], 8] = 1
    data[game_state['self'][3][0], game_state['self'][3][1], 9] = int(game_state['self'][2])

    for other in game_state['others']:
        data[other[3][0], other[3][1], 10] = 1
        data[other[3][0], other[3][1], 11] = int(other[2])
    
    return data

def setup(self):
    self.model = load_model('model')
    self.is_training = False

def act(agent, game_state: dict):
    r = game_state['round']
    if agent.is_training and r < 0:
        dist_action = np.array([1, 1, 1, 1, 0.2, 0])

    else:
        features = to_features(game_state)
        features = features.reshape([1,*features.shape])

        dist_action = np.clip(agent.model.predict(features)[0], 0, 100)
        dist_action = np.nan_to_num(dist_action)+5e-2

    dist_action /= np.sum(dist_action)
    action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=dist_action)
    #action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'][np.argmax(dist_action)]
    agent.logger.info('Pick %s'%action)
    return action
