import numpy as np
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.keras.metrics import mean_absolute_error
import events as e
import matplotlib.pyplot as plt
import tensorflow as tf

def masked_mean_squared_error(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, 0))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, 0))
    return mean_squared_error(y_true_masked, y_pred_masked)

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

def select_features(X):
    X_new = np.zeros((7, 7, 12))
    data = np.pad(X, pad_width=((3,3), (3,3), (0,0)))
    coords = np.where(data[8]==1)[0]
    X_new = data[coords[0]-3:coords[0]+4, coords[1]-3:coords[1]+4]
    return X_new

def to_data(game_state):
    data2 = np.array([game_state['round'],
                     game_state['step'],
                     game_state['self'][1]])
    return data2

def to_labels(action):
    action_array = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    data = np.zeros(6)
    idx = np.where(action_array==action)
    data[idx] = 1
    return data

event_index = np.array([e.WAITED, e.INVALID_ACTION, e.BOMB_DROPPED, e.BOMB_EXPLODED, e.CRATE_DESTROYED, e.COIN_FOUND, e.COIN_COLLECTED, e.KILLED_OPPONENT, e.KILLED_SELF, e.SURVIVED_ROUND])
def to_rewards(events):
    data = np.zeros(len(event_index))
    for ev in events:
        data[np.where(event_index==ev)] += 1
    
    return data

def build_model():
    model = Sequential()
    model.add(Input(shape=(7, 7, 12)))
    model.add(Conv2D(4, 1, activation='relu', padding='same'))
    model.add(Flatten())
    #model.add(GlobalAveragePooling2D())
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.compile(optimizer=Adam(lr=3e-4), loss=masked_mean_squared_error, metrics=['accuracy'])

    model.summary()
    return model

def setup_training(self):
    self.model = load_model('model', custom_objects={'masked_mean_squared_error':masked_mean_squared_error})#build_model()
    self.reward = []
    self.features = []
    self.labels = []

def game_events_occurred(self, old_game_state, action, new_game_state, events):
    if old_game_state != None:
        reward = np.sum(to_rewards(events)*np.array([0, 0, 0, 0, 0.5, 0.5, 1, 1, -1, 0]))
        self.logger.info('reward:'+str(to_rewards(events)))
        self.reward.append(0)
        self.features.append(select_features(to_features(old_game_state)))
        self.labels.append(to_labels(action))
        if len(self.reward) > 1:
            self.reward[-2] = reward

def end_of_round(self, last_game_state, last_action, events):
    self.logger.info(self.reward)
    features = np.array(self.features)
    labels = np.array(self.labels)*np.array(self.reward)[:,None]
    idx_success = np.where(np.array(self.reward)!=0)
    self.logger.info(idx_success)
    features = features[idx_success]
    labels = labels[idx_success]
    if np.any(np.array(self.reward)!=0):
        self.model.fit(features, labels, epochs=4, batch_size=4)
    self.features = []
    self.labels = []
    self.reward = []
    self.model.save('model')
