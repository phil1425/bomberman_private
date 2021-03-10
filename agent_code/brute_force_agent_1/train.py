import events as e
import numpy as np
import h5py

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

def to_labels(action):
    action_array = np.array(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'])
    data = np.zeros(6)
    idx = np.where(action_array==action)
    data[idx] = 1
    return data

def setup_training(self):
    file_str = int(np.random.random()*1e10)
    self.data = h5py.File('data_%s.h5'%file_str, 'w')
    self.data.create_dataset('features', (100, 17, 17, 12), maxshape=(None, 17, 17, 12), chunks=True)
    self.data.create_dataset('labels', (100, 6), maxshape=(None, 6), chunks=True)
    self.data.create_dataset('score', (100, ), maxshape=(None, ), chunks=True)
    self.curr_size = 0
    self.curr_space = 100
    self.list_features = []
    self.list_labels = []


def game_events_occurred(self, old_game_state, action, new_game_state, events):
    if old_game_state != None:
        self.list_features.append(to_features(old_game_state))
        self.list_labels.append(to_labels(action))

def end_of_round(self, last_game_state, last_action, events):
    score = last_game_state['self'][1]
    avail = self.curr_space-self.curr_size
    if avail >= len(self.list_labels):
        self.data['features'].resize((self.curr_space+200, 17, 17, 12))
        self.data['labels'].resize((self.curr_space+200, 6))
        self.data['score'].resize((self.curr_space+200, ))
        self.curr_space += 200

    new_size = self.curr_size+len(self.list_labels)

    self.data['features'][self.curr_size:new_size] = np.array(self.list_features)
    self.data['labels'][self.curr_size:new_size] = np.array(self.list_labels)
    self.data['score'][self.curr_size:new_size] = np.full(len(self.list_labels), score)
    self.curr_size = new_size
    self.list_features = []
    self.list_labels = []
