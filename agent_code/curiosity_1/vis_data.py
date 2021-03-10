import numpy as np
import matplotlib.pyplot as plt

old = np.load('agent_code/curiosity_1/old.npy')
new = np.load('agent_code/curiosity_1/new.npy')
act = np.load('agent_code/curiosity_1/act.npy')
actions = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])

def select_features(X):
    X_new = np.zeros([X.shape[0], 7, 7, 12])
    for i in range(X.shape[0]):
        data = np.pad(X[i], pad_width=((3,3), (3,3), (0,0)))
        coords = np.array(np.where(data[:,:,8]==1))[:,0]
        print(coords)
        X_new[i] = data[coords[0]-3:coords[0]+4, coords[1]-3:coords[1]+4]
    return X_new

old_sel = select_features(old)
new_sel = select_features(new)

print(old.shape, new.shape, act.shape)
for i in range(200):
    action = actions[np.argmax(act[i])]
    print(i, action)
    for j in [0,1,6,8]:
        plt.figure()
        plt.subplot(221)
        plt.imshow(old[i,:,:,j])
        plt.subplot(222)
        plt.imshow(new[i,:,:,j])
        plt.subplot(223)
        plt.imshow(old_sel[i,:,:,j])
        plt.subplot(224)
        plt.imshow(new_sel[i,:,:,j])
        plt.show()

