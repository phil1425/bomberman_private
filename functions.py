import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error, mae
from treesearch import MovementTree

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

def handmade_features(game_state):
    data = []#np.zeros(64)
    coords = game_state['self'][3]
    x = coords[0]
    y = coords[1]
    data.append(game_state['step'])
    data.append(game_state['self'][1])
    data.append(int(game_state['self'][2]))

    #walls
    walls = (game_state['field']==-1).astype(int)
    walls[0,:] += 1
    walls[-1,:] += 1
    walls[:,0] += 1
    walls[:,-1] += 1
    data.append(walls[x, y+1])
    data.append(walls[x, y-1])
    data.append(walls[x+1, y])
    data.append(walls[x-1, y])
    #agent.logger.info('walls:'+str(data[-4:]))

    #crates
    crates = (game_state['field']==1).astype(int)
    crates_1 = np.pad(crates, 2)
    crates_1 = crates_1[x:x+5, y:y+5].reshape(5*5)
    mask = np.full(5*5, True)
    mask[13] = False
    data.extend(crates_1[mask])
    #agent.logger.info('crates1:'+str(crates_1))

    crates_2 = np.pad(crates, 2)
    crates_2[x:x+5, y:y+5] = np.zeros((5,5))
    crates_3 = np.zeros(8)
    crates_3[0] = np.sum(crates_2[x+2, :y+2])
    crates_3[1] = np.sum(crates_2[x+2, y+3:])
    crates_3[2] = np.sum(crates_2[x+3:, y+2])
    crates_3[3] = np.sum(crates_2[:x+2, y+2])

    crates_3[4] = np.sum(crates_2[x+3:, y+3:])
    crates_3[5] = np.sum(crates_2[:x+2, y+3:])
    crates_3[6] = np.sum(crates_2[x+3:, :y+2])
    crates_3[7] = np.sum(crates_2[:x+2, :y+2])
    data.extend(crates_3)
    #agent.logger.info('crates3:'+str(crates_3))

    bomb_map = np.zeros((17+6, 17+6))
    bomb_cooldowns = [x[1] for x in game_state['bombs']]
    bomb_idx_sorted = np.flip(np.argsort(bomb_cooldowns))

    for idx in bomb_idx_sorted:
        bomb = game_state['bombs'][idx]
        b_x = bomb[0][0]
        b_y = bomb[0][1]
        bomb_map[b_x:b_x+7, b_y+3] = bomb[1]
        bomb_map[b_x+3, b_y:b_y+7] = bomb[1]
    #agent.logger.info('bombs__map:'+str(bomb_map))

    bombs_x = bomb_map[x+3, y:y+7]
    bombs_y = bomb_map[x:x+7, y+3]
    #agent.logger.info('bombs_x:'+str(bombs_x))
    #agent.logger.info('bombs_y:'+str(bombs_y))

    data.extend(bombs_x)
    data.extend(bombs_y)

    c_coin = [0, 0]
    data_dist_coin = []
    for coin in game_state['coins']:
        dist = np.sqrt(np.sum(np.square([coin[0]-x, coin[1]-y])))
        data_dist_coin.append(dist)
    if data_dist_coin != []:
        c_coin[0] = x-game_state['coins'][np.argmin(data_dist_coin)][0]
        c_coin[1] = y-game_state['coins'][np.argmin(data_dist_coin)][1]

    data.extend(c_coin)
    #agent.logger.info('coin:'+str(c_coin))

    c_agent = [0, 0]
    data_dist_agent = []
    for agent in game_state['others']:
        dist = np.sqrt(np.sum(np.square([agent[3][0]-x, agent[3][1]-y])))
        data_dist_agent.append(dist)
    if data_dist_agent != []:
        c_agent[0] = x-game_state['others'][np.argmin(data_dist_agent)][3][0]
        c_agent[1] = y-game_state['others'][np.argmin(data_dist_agent)][3][1]

    data.extend(c_agent)
    #agent.logger.info('agent:'+str(c_agent))

    return np.array(data)

def minimal_features(game_state):
    data = []#np.zeros(64)
    coords = game_state['self'][3]
    x = coords[0]
    y = coords[1]
    data.append(int(game_state['self'][2]))

    #walls
    walls = (game_state['field']==-1).astype(int)
    data.append(walls[x, y+1])
    data.append(walls[x, y-1])
    data.append(walls[x+1, y])
    data.append(walls[x-1, y])
    return np.array(data)

def minimal_features_2(game_state):
    data = []#np.zeros(64)
    coords = game_state['self'][3]
    x = coords[0]
    y = coords[1]
    data.append(int(game_state['self'][2]))
    data.append(int(game_state['step']))

    walls = game_state['field']
    tree = MovementTree((x, y), game_state['field'], 7)
    data.extend(tree.get_free_spaces())

    bomb_map = np.zeros((17+6, 17+6))
    bomb_cooldowns = [x[1] for x in game_state['bombs']]
    bomb_idx_sorted = np.flip(np.argsort(bomb_cooldowns))
    for idx in bomb_idx_sorted:
        bomb = game_state['bombs'][idx]
        b_x = bomb[0][0]
        b_y = bomb[0][1]

        bomb_map[b_x+3, b_y+3] = bomb[1]

        if walls[b_x, b_y+1] >= 0:
            bomb_map[b_x+3, b_y+4:b_y+7] = bomb[1]
        if walls[b_x, b_y-1] >= 0:
            bomb_map[b_x+3, b_y:b_y+2] = bomb[1]
        if walls[b_x+1, b_y] >= 0:
            bomb_map[b_x+4:b_x+7, b_y+3] = bomb[1]
        if walls[b_x-1, b_y] >= 0:
            bomb_map[b_x:b_x+3, b_y+3] = bomb[1]

    bombs_x = bomb_map[x+3, y+2:y+5]
    bombs_y = bomb_map[[x+2,x+5], y+3]
    data.extend(bombs_x)
    data.extend(bombs_y)

    crates = (game_state['field']==1).astype(int)
    crates_1 = np.pad(crates, 3)
    num_explodable_crates = 0
    if walls[x, y+1] < 0:
        num_explodable_crates += np.sum(crates_1[x+3, y+4:y+7])
    if walls[x, y-1] < 0:
        num_explodable_crates += np.sum(crates_1[x+3, y:y+3])
    if walls[x+1, y] < 0:
        num_explodable_crates += np.sum(crates_1[x+4:x+7, y+3])
    if walls[x-1, y] < 0:
        num_explodable_crates += np.sum(crates_1[x:x+3, y+3])
    data.append(num_explodable_crates)

    c_coin = [0, 0]
    data_dist_coin = []
    for coin in game_state['coins']:
        dist = np.sqrt(np.sum(np.square([coin[0]-x, coin[1]-y])))
        data_dist_coin.append(dist)
    if data_dist_coin != []:
        c_coin[0] = x-game_state['coins'][np.argmin(data_dist_coin)][0]
        c_coin[1] = y-game_state['coins'][np.argmin(data_dist_coin)][1]
    data.extend(c_coin)

    c_agent = [0, 0]
    data_dist_agent = []
    for agent in game_state['others']:
        dist = np.sqrt(np.sum(np.square([agent[3][0]-x, agent[3][1]-y])))
        data_dist_agent.append(dist)
    if data_dist_agent != []:
        c_agent[0] = x-game_state['others'][np.argmin(data_dist_agent)][3][0]
        c_agent[1] = y-game_state['others'][np.argmin(data_dist_agent)][3][1]
    data.extend(c_agent)
    
    return np.array(data)


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

    for coin in game_state['coins']:
        data[coin[0], coin[1], 6] = 1

    data[game_state['self'][3][0], game_state['self'][3][1], 7] = 1

    for other in game_state['others']:
        data[other[3][0], other[3][1], 8] = 1
    
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

def augment_game_states(game_state_list, action_list):
    augmented_states = []
    augmented_actions = []
    for game_state, action in zip(game_state_list, action_list):
        for mirror in [False, True]:
            game_state_mirrored = game_state.copy()
            action_mirrored = action.copy()
            if mirror:
                game_state_mirrored['field'] = np.flip(game_state['field'], axis=-1)
                game_state_mirrored['bombs'] = [((b[0][0], 16-b[0][1]), b[1]) for b in game_state['bombs']]
                game_state_mirrored['explosion_map'] = np.flip(game_state['explosion_map'], axis=-1)
                game_state_mirrored['coins'] = [(b[0], 16-b[1]) for b in game_state['coins']]
                b = game_state_mirrored['self']
                game_state_mirrored['self'] = (b[0], b[1], b[2], (b[3][0], 16-b[3][1]))
                game_state_mirrored['others'] = [(b[0], b[1], b[2], (b[3][0], 16-b[3][1])) for b in game_state['others']]

                action_mirrored = action[[1, 0, 2, 3, 4, 5]]

            for rot in [0, 1, 2, 3]:
                game_state_rotated = game_state_mirrored.copy()
                action_rotated = action_mirrored.copy()

                for _ in range(rot):
                    game_state_rotated['field'] = np.rot90(game_state_rotated['field'], 3)
                    game_state_rotated['bombs'] = [((b[0][1], 16-b[0][0]), b[1]) for b in game_state_rotated['bombs']]
                    game_state_rotated['explosion_map'] = np.rot90(game_state_rotated['explosion_map'], 3)
                    game_state_rotated['coins'] = [(b[1], 16-b[0]) for b in game_state_rotated['coins']]
                    b = game_state_rotated['self']
                    game_state_rotated['self'] = (b[0], b[1], b[2], (b[3][1], 16-b[3][0]))
                    game_state_rotated['others'] = [(b[0], b[1], b[2], (b[3][1], 16-b[3][0])) for b in game_state_rotated['others']]

                    action_rotated = action_rotated[[3, 2, 0, 1, 4, 5]]
                
                augmented_states.append(game_state_rotated)
                augmented_actions.append(action_rotated)
    
    return augmented_states, augmented_actions

def rotate_game_states(game_state_list, action_list):
    augmented_states = []
    augmented_actions = []
    for game_state, action in zip(game_state_list, action_list):
        mirror = True

        mirror_map = np.array([
            [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0],
            [1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0],
            [1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0],
            [1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0],
            [1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0],
            [1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1],
            [0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1],
            [0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1],
            [0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1],
            [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1],
            [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        ])

        rotation_map = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3],
            [1,1,0,0,0,0,0,0,0,0,0,0,0,0,3,3,3],
            [1,1,1,0,0,0,0,0,0,0,0,0,0,3,3,3,3],
            [1,1,1,1,0,0,0,0,0,0,0,0,3,3,3,3,3],
            [1,1,1,1,1,0,0,0,0,0,0,3,3,3,3,3,3],
            [1,1,1,1,1,1,0,0,0,0,3,3,3,3,3,3,3],
            [1,1,1,1,1,1,1,0,0,3,3,3,3,3,3,3,3],
            [1,1,1,1,1,1,1,1,0,3,3,3,3,3,3,3,3],
            [1,1,1,1,1,1,1,1,2,2,3,3,3,3,3,3,3],
            [1,1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,3],
            [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3],
            [1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3],
            [1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3],
            [1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3],
            [1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3],
            [1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        ])

        x = game_state['self'][3][0]
        y = game_state['self'][3][1]
        mirror = bool(mirror_map[x, y])
        rot = rotation_map[x, y]

        game_state_rotated = game_state.copy()
        action_rotated = action.copy()

        for _ in range(rot):
            game_state_rotated['field'] = np.rot90(game_state_rotated['field'], 3)
            game_state_rotated['bombs'] = [((b[0][1], 16-b[0][0]), b[1]) for b in game_state_rotated['bombs']]
            game_state_rotated['explosion_map'] = np.rot90(game_state_rotated['explosion_map'], 3)
            game_state_rotated['coins'] = [(b[1], 16-b[0]) for b in game_state_rotated['coins']]
            b = game_state_rotated['self']
            game_state_rotated['self'] = (b[0], b[1], b[2], (b[3][1], 16-b[3][0]))
            game_state_rotated['others'] = [(b[0], b[1], b[2], (b[3][1], 16-b[3][0])) for b in game_state_rotated['others']]

            action_rotated = action_rotated[[3, 2, 0, 1, 4, 5]]

        game_state_mirrored = game_state_rotated.copy()
        action_mirrored = action_rotated.copy()
        if mirror:
            game_state_mirrored['field'] = np.flip(game_state_mirrored['field'], axis=-1)
            game_state_mirrored['bombs'] = [((b[0][0], 16-b[0][1]), b[1]) for b in game_state_mirrored['bombs']]
            game_state_mirrored['explosion_map'] = np.flip(game_state_mirrored['explosion_map'], axis=-1)
            game_state_mirrored['coins'] = [(b[0], 16-b[1]) for b in game_state_mirrored['coins']]
            b = game_state_mirrored['self']
            game_state_mirrored['self'] = (b[0], b[1], b[2], (b[3][0], 16-b[3][1]))
            game_state_mirrored['others'] = [(b[0], b[1], b[2], (b[3][0], 16-b[3][1])) for b in game_state_mirrored['others']]

            action_mirrored = action_mirrored[[1, 0, 2, 3, 4, 5]]
        

        augmented_states.append(game_state_mirrored)
        augmented_actions.append(action_mirrored)
    
    return augmented_states, augmented_actions, rot, mirror

def augment_labels(labels):
    augmented_labels = []
    for label in labels:
        for mirror in [False, True]:
            mirrored_label = label.copy()
            if mirror:
                mirrored_label = mirrored_label[[1, 0, 2, 3, 4, 5]]
            for rot in [0, 1, 2, 3]:
                rotated_label = mirrored_label.copy()
                for _ in range(rot):
                    rotated_label = rotated_label[[3, 2, 0, 1, 4, 5]]
                augmented_labels.append(rotated_label)
    return np.array(augmented_labels)

def reverse_rotate_action(label, rot, mirror):
    if rot == 3:
        rot = 1
    elif rot == 1:
        rot = 3

    mirrored_label = label.copy()
    if mirror:
        mirrored_label = mirrored_label[[1, 0, 2, 3, 4, 5]]
    rotated_label = mirrored_label.copy()
    for _ in range(rot):
        rotated_label = rotated_label[[3, 2, 0, 1, 4, 5]]

    return rotated_label

if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt
    states = pickle.load(open('featureset.pt', 'rb'))

    for state in states:
        plt.figure()
        vis_field = state['field'].copy()
        x, y = state['self'][3]
        vis_field[x, y] = 3

        for bomb in state['bombs']:
            bx, by = bomb[0]
            vis_field[bx, by] = 2
        plt.imshow(vis_field)
        print(minimal_features_2(state))
        plt.show()
    

