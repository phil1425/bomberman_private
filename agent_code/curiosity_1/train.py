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

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def masked_mean_squared_error(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, 0))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, 0))
    return mean_squared_error(y_true_masked, y_pred_masked)

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
    new_labels[1] = labels[:,[0, 1, 3, 2, 4, 5]]
    new_labels[2] = labels[:,[1, 0, 2, 3, 4, 5]]
    new_labels[3] = labels[:,[1, 0, 3, 2, 4, 5]]
    #new_labels[4] = labels[[0, 1, 2, 3, 4, 5]]
    #new_labels[5] = labels[[0, 1, 2, 3, 4, 5]]
    #new_labels[6] = labels[[0, 1, 2, 3, 4, 5]]

    return np.concatenate(new_features, axis=0), np.concatenate(new_labels, axis=0)

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
    X_new = np.zeros([X.shape[0], 7, 7, 12])
    for i in range(X.shape[0]):
        data = np.pad(X[i], pad_width=((3,3), (3,3), (0,0)))
        coords = np.array(np.where(data[:,:,8]==1))[:,0]
        #print(coords)
        X_new[i] = data[coords[0]-3:coords[0]+4, coords[1]-3:coords[1]+4]
    return X_new

def to_labels(action):
    action_array = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    data = np.zeros(6)
    idx = np.where(action_array==action)
    data[idx] = 1
    return data

encoding_dim = 32

def build_encoder():
    model = Sequential()
    model.add(Input(shape=(17, 17, 12)))
    model.add(Conv2D(8, 3, activation='relu', padding='same'))
    model.add(Conv2D(8, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(8, 1, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(encoding_dim, activation='relu'))

    model.summary()
    return model

def build_inverse(encoder):
    old_state = Input(shape=(17,17,12))
    new_state = Input(shape=(17,17,12))

    old_encoding = encoder(old_state)
    new_encoding = encoder(new_state)

    x = Concatenate(axis=-1)([old_encoding, new_encoding])
    x = Dense(32, activation='relu')(x)
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

def build_policy():
    model = Sequential()
    model.add(Input(shape=(17, 17, 12)))
    model.add(Conv2D(4, 1, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))

    model.summary()
    return model

def setup_training(self):
    self.encoder = build_encoder()
    self.inverse = build_inverse(self.encoder)
    self.forward = build_forward()
    self.policy = build_policy()

    self.inverse.compile(Adam(3e-4), loss=categorical_crossentropy, metrics=['accuracy'])
    self.forward.compile(Adam(3e-4), loss=mse)
    self.policy.compile(Adam(3e-4), loss=masked_mean_squared_error)

    new_input = Input(shape=(64))
    x = self.inverse.layers[4](new_input)
    new_output = self.inverse.layers[5](x)
    self.new_inverse = Model(inputs=[new_input], outputs=[new_output])

    self.new_features = []
    self.old_features = []
    self.labels = []
    self.num_invalid_moves = 0

def game_events_occurred(self, old_game_state, action, new_game_state, events):
    self.logger.info(events)
    if e.INVALID_ACTION in events:
        self.num_invalid_moves += 1
    if old_game_state != None and new_game_state != None:
        self.old_features.append(to_features(old_game_state))
        self.new_features.append(to_features(new_game_state))
        self.labels.append(to_labels(action))

def end_of_round(self, last_game_state, last_action, events):
    self.new_features.append(to_features(last_game_state))
    self.old_features.append(self.new_features[-1])
    self.labels.append(to_labels(last_action))

    new_features = np.array(self.new_features)
    old_features = np.array(self.old_features)
    labels = np.array(self.labels)

    self.logger.info('invalid moves %f'%(self.num_invalid_moves/len(self.labels)))

    new_features_aug, labels_aug = augment(new_features, labels)
    old_features_aug, _ = augment(old_features, labels)

    #np.save('new.npy', new_features_aug)
    #np.save('old.npy', old_features_aug)
    #np.save('act.npy', labels_aug)

    self.inverse.fit([new_features_aug, old_features_aug], labels_aug, batch_size=4, epochs=4)
    theta_0 = self.encoder.predict(old_features_aug)
    theta_1 = self.encoder.predict(new_features_aug)

    X_forward = np.concatenate([theta_0, labels_aug], axis=-1)

    self.forward.fit(X_forward, theta_1, batch_size=4, epochs=4)

    theta_pred = self.forward.predict(X_forward)
    reward = np.sum(np.square(theta_1-theta_pred), axis=-1)

    y_policy = labels_aug*reward[:,None]

    X_policy = select_features(old_features_aug)
    self.policy.fit(X_policy, y_policy, batch_size=4, epochs=4)

    self.new_features = []
    self.old_features = []
    self.labels = []

    self.num_invalid_moves = 0
    if last_game_state['round'] % 10 == 0: 
        self.encoder.save('encoder')
        self.inverse.save('inverse')
        self.forward.save('forward')
        self.policy.save('policy')
