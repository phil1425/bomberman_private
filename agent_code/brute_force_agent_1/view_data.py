import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt

fpaths = glob.glob('agent_code/brute_force_agent_1/data_*.h5')
list_features = []
list_labels = []



for fpath in fpaths:
    f = h5py.File(fpaths[0], 'r')
    #features = np.array(f['features'])
    #labels = np.array(f['labels'])
    score = np.array(f['score'])
    idx_success = np.where(score==1)

    features_success = np.array(f['features'][idx_success])
    labels_success = np.array(f['labels'][idx_success])
    list_features.append(features_success)
    list_labels.append(labels_success)

features = np.concatenate(list_features, axis=0)
labels = np.concatenate(list_labels, axis=0)

np.save('agent_code/brute_force_agent_1/features_success.npy', features)
np.save('agent_code/brute_force_agent_1/labels_success.npy', labels)

plt.figure()
plt.hist(score)
plt.show()

print(np.sum(labels, axis=0))

#features = f['features']
#labels = f['labels']
#score = f['score']
#for i in range(10):
#    print(features[i])
#    print(labels[i])
#    print(score[i])