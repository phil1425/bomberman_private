import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error, mae

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

def masked_mean_squared_error(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, 0))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, 0))
    return mean_squared_error(y_true_masked, y_pred_masked)

def masked_mae(y_true, y_pred):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, 0))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, 0))
    return mae(y_true_masked, y_pred_masked)

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