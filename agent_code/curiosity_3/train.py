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
from functions import augment, select_features, masked_mean_squared_error, to_features, to_labels, masked_mae, handmade_features
from tree_model import TreeModel

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

encoding_dim = 64
def build_encoder():
    model = Sequential()
    model.add(Input(shape=(17, 17, 12)))
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

def setup_training(self):
    if False:
        #self.policy = load_model('policy', custom_objects={'masked_mean_squared_error':masked_mean_squared_error})
        self.policy = TreeModel(4000, (64, ), (6, ))
        self.policy.load('policy.p')

        self.inverse = load_model('inverse')
        self.inverse.summary()
        new_input = Input(shape=(17, 17, 12))
        new_output = self.inverse.layers[2](new_input)
        self.encoder = Model(inputs=[new_input], outputs=[new_output])
        self.encoder.summary()

        self.forward = load_model('forward')
    
    else:
        self.encoder = build_encoder()
        self.inverse = build_inverse(self.encoder)
        self.forward = build_forward()
        self.policy = TreeModel(2000, (56, ), (6, ), 10)

        self.inverse.compile(Adam(3e-4), loss=categorical_crossentropy, metrics=['accuracy'])
        self.forward.compile(Adam(3e-4), loss=mse)

    self.new_features = []
    self.old_features = []
    self.old_policy_features = []
    self.new_policy_features = []
    self.labels = []
    self.num_invalid_moves = 0

def game_events_occurred(self, old_game_state, action, new_game_state, events):
    self.logger.info(events)
    if e.INVALID_ACTION in events:
        self.num_invalid_moves += 1
    if old_game_state != None and new_game_state != None:
        self.old_features.append(to_features(old_game_state))
        self.new_features.append(to_features(new_game_state))

        self.old_policy_features.append(handmade_features(old_game_state))
        self.new_policy_features.append(handmade_features(new_game_state))

        self.labels.append(to_labels(action))

def end_of_round(self, last_game_state, last_action, events):
    self.new_features.append(to_features(last_game_state))
    self.new_policy_features.append(handmade_features(last_game_state))

    self.old_features.append(self.new_features[-2])
    self.old_policy_features.append(self.new_policy_features[-2])

    self.labels.append(to_labels(last_action))

    new_features = np.array(self.new_features)
    old_features = np.array(self.old_features)
    old_policy_features = np.array(self.old_policy_features)
    new_policy_features = np.array(self.new_policy_features)
    num_steps = old_policy_features.shape[0]

    labels = np.array(self.labels)

    self.logger.info('invalid moves %f'%(self.num_invalid_moves/len(self.labels)))

    new_features_aug, labels_aug = augment(new_features, labels)
    old_features_aug, _ = augment(old_features, labels)

    #np.save('new.npy', new_features_aug)
    #np.save('old.npy', old_features_aug)
    #np.save('act.npy', labels_aug)

    self.inverse.fit([new_features_aug, old_features_aug], labels_aug, batch_size=4, epochs=1)
    theta_0 = self.encoder.predict(old_features_aug)
    theta_1 = self.encoder.predict(new_features_aug)

    X_forward = np.concatenate([theta_0, labels_aug], axis=-1)

    self.forward.fit(X_forward, theta_1, batch_size=4, epochs=1)

    theta_pred = self.forward.predict(X_forward[:num_steps])
    reward = np.sum(np.square(theta_1[:num_steps]-theta_pred), axis=-1)

    y_policy = labels*reward[:,None]
    X_policy = old_policy_features#.reshape([old_features_aug.shape[0], 17*17*12])
    self.policy.fit(X_policy, y_policy)

    self.new_features = []
    self.old_features = []
    self.old_policy_features = []
    self.new_policy_features = []
    self.labels = []

    self.num_invalid_moves = 0
    if last_game_state['round'] % 100 == 0: 
        self.encoder.save('encoder')
        self.inverse.save('inverse')
        self.forward.save('forward')
        self.policy.save('policy.p')
