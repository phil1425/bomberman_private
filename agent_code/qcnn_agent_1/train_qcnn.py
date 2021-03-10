import numpy as np
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
from tensorflow.keras.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def build_model():
    model = Sequential()
    model.add(Input(shape=(17, 17, 12)))
    #model.add(Conv2D(4, 1, activation='relu', padding='same'))
    #model.add(Flatten())
    model.add(Conv2D(16, 1, activation='relu', padding='same'))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(Conv2D(16, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(MaxPool2D(2))
    model.add(Conv2D(32, 3, activation='relu', padding='same'))
    model.add(Conv2D(8, 1, activation='relu', padding='same'))
    model.add(Flatten())
    #model.add(GlobalAveragePooling2D())
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.compile(optimizer=Adam(lr=3e-4), loss=mean_squared_error, metrics=['accuracy'])

    model.summary()
    return model

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

labels = np.load('agent_code/qcnn_agent_1/labels_success.npy')
features = np.load('agent_code/qcnn_agent_1/features_success.npy')
print(labels.shape, features.shape)
features, labels = augment(features, labels)

model = build_model()
model.fit(features, labels, epochs=16, batch_size=4, validation_split=0.1)
model.save('agent_code/qcnn_agent_1/model')