import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt

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

event_index = np.array(['WAITED', 'INVALID_ACTION', 'BOMB_DROPPED', 'BOMB_EXPLODED', 'CRATE_DESTROYED', 'COIN_FOUND', 'COIN_COLLECTED', 'KILLED_OPPONENT', 'KILLED_SELF', 'SURVIVED_ROUND'])
reward_mask = np.array([-1, -1, 0, 0, 0.5, 0.5, 1, 5, -1, 0])

def per_step_aux_reward(rewards):
    kernel = np.array([0.1, 0.3,0.6,1,0,0,0])
    rewards *= reward_mask
    conv_rewards = np.zeros(rewards.shape)
    for i in range(len(reward_mask)):
        conv_rewards[:,i] = np.convolve(rewards[:,i], kernel, 'same')
    return np.sum(conv_rewards, axis=1)


fpaths = glob.glob('agent_code/offline_agent_1/data_*.h5')

list_features = []
list_labels = []

for fpath in fpaths:
    f = h5py.File(fpath, 'r')
    score = np.array(f['score'])
    rewards = np.array(f['rewards'])
    aux_reward = per_step_aux_reward(rewards)
    features = np.array(f['features'])

    idx_success = np.where(np.logical_and(np.logical_or(aux_reward!=0, score>=0), np.any(features[:,:,:,8], axis=(1,2))))

    features = np.array(f['features'][idx_success])
    aux_reward = aux_reward[idx_success]
    score = score[idx_success]
    labels = np.array(f['labels'][idx_success])

    print(labels.shape, score.shape, rewards.shape, features.shape)
    #aux_reward = per_step_aux_reward(rewards)
    print(aux_reward.shape)

    labels_inv = (labels).copy()-1

    labels = labels*((score/20)+aux_reward)[:, None]
    #labels -= labels_inv*np.mean(score)

    list_features.append(features)
    list_labels.append(labels)

features = np.concatenate(list_features, axis=0)
labels = np.concatenate(list_labels, axis=0)
print(labels)

np.save('agent_code/offline_agent_1/features.npy', features)
np.save('agent_code/offline_agent_1/labels.npy', labels)

