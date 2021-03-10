import numpy as np
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.keras.metrics import mean_absolute_error
import events as e
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

def augment(features, labels):
    new_features = np.zeros([4, *features.shape])
    new_labels = np.zeros([4, *labels.shape])

    new_features[0] = features
    new_features[1] = np.flip(features, axis=1)
    new_features[2] = np.flip(features, axis=2)
    new_features[3] = np.flip(features, axis=(1,2))
    #new_features[4] = np.rot90(features, axes=(1,2))
    #new_features[5] = np.rot90(new_features[4], axes=(1,2))
    #new_features[6] = np.rot90(new_features[5], axes=(1,2))

    new_labels[0] = labels
    new_labels[1] = labels[:,[1, 0, 2, 3, 4, 5]]
    new_labels[2] = labels[:,[0, 1, 3, 2, 4, 5]]
    new_labels[3] = labels[:,[1, 0, 3, 2, 4, 5]]
    #new_labels[4] = labels[[0, 1, 2, 3, 4, 5]]
    #new_labels[5] = labels[[0, 1, 2, 3, 4, 5]]
    #new_labels[6] = labels[[0, 1, 2, 3, 4, 5]]

    return np.concatenate(new_features, axis=0), np.concatenate(new_labels, axis=0)

def to_labels(action):
    action_array = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    data = np.zeros(6)
    idx = np.where(action_array==action)
    data[idx] = 1
    return data

def build_model():
    model = Sequential()
    model.add(Input(shape=(17, 17, 12)))
    #model.add(Conv2D(4, 1, activation='relu', padding='same'))
    #model.add(Flatten())
    model.add(Conv2D(8, 1, activation='relu', padding='same'))
    model.add(Conv2D(8, 3, activation='relu', padding='same'))
    model.add(Conv2D(8, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(8, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(8, 3, activation='relu', padding='same'))
    model.add(Flatten())
    #model.add(GlobalAveragePooling2D())
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.compile(optimizer=Adam(lr=3e-4), loss=mean_squared_error, metrics=['accuracy'])

    model.summary()
    return model

def setup_training(agent):
    agent.features = []
    agent.labels = []
    agent.model = load_model('model') #build_model()
    agent.version = np.loadtxt('v.txt')
    agent.aux_reward = 0
    agent.is_training = True

def game_events_occurred(agent, old_game_state, action, new_game_state, events):
    if e.INVALID_ACTION in events:
       agent.aux_reward += 0

    if not (e.INVALID_ACTION in events):
       agent.aux_reward += 0

    if e.WAITED in events:
       agent.aux_reward += 0

    if e.COIN_FOUND in events:
       agent.aux_reward += 0.1

    if e.CRATE_DESTROYED in events:
       agent.aux_reward += 0.01
    
    if old_game_state != None:
        agent.features.append(to_features(old_game_state))
        agent.labels.append(to_labels(action))
    agent.logger.info('aux reward is %f'%agent.aux_reward)

def end_of_round(agent, last_game_state, last_action, events):
    reward = last_game_state['self'][1]+agent.aux_reward
    a = all([last_game_state['self'][1] > a[1] for a in last_game_state['others']])
    b = all([np.sum(last_game_state['self'][3]) > np.sum(a[3]) for a in last_game_state['others']])
    winner = (a and reward > 0) or (b and np.random.random()>0.95)

    if len(agent.features) > 1 and winner:
        features, labels = augment(np.array(agent.features), np.array(agent.labels))
        expected = agent.model.predict(features)
        expected[labels==1] = reward
        agent.model.fit(features, expected, epochs=32, batch_size=4)
        agent.model.save('model')
        agent.version += 1
        np.savetxt('v.txt', [agent.version])

        #agent.model.summary()

    agent.features = []
    agent.labels = []
    agent.aux_reward = 0
    if not winner:
        v = np.loadtxt('v.txt')
        if agent.version < v:
            agent.model = load_model('model')
            agent.version = v
