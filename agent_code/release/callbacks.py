import numpy as np
from sklearn import tree
import pickle

class TreeModel():
    def __init__(self, buffer_size, X_shape, y_shape, min_samples=1, max_leaf_nodes=None):
        self.buffer_X = CircularArray((buffer_size, *X_shape))
        self.buffer_y = CircularArray((buffer_size, *y_shape))
        self.model = tree.DecisionTreeRegressor(min_samples_leaf=min_samples, max_leaf_nodes=None)
        self.fitted = False
    
    def fit(self, X, y, lr=1):
        for i in range(X.shape[0]):
            self.buffer_X.add(X[i])
            self.buffer_y.add(y[i])

        data_X = self.buffer_X.as_array()
        data_y = self.buffer_y.as_array().copy()

        if self.fitted:
            data_pred = self.predict(data_X)
            zeros = np.where(data_y==0)
            data_y[data_y!=0] -= data_pred[data_y!=0]
            data_y[data_y!=0] *= lr
            data_y[data_y!=0] += data_pred[data_y!=0]
            data_y[zeros] = data_pred[zeros]

        else:
            data_pred = np.mean(data_y, axis=0)
            zeros = np.where(data_y==0)
            data_y[zeros] = data_pred[zeros[1]]

        self.model.fit(data_X, data_y)
        self.fitted = True

    def add_data(self, X, y):
        for i in range(X.shape[0]):
            self.buffer_X.add(X[i])
            self.buffer_y.add(y[i])
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        pickle.dump(self.model, open(path, "wb"))

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))

class Node():
    pass

class MovementTree():
    def __init__(self, agent_coords, field, max_free_spaces):
        self.root = Node()
        self.root.direction = -1
        self.root.coords = agent_coords

        self.field = (field != 0).astype(int)
        self.visited_spaces = [agent_coords]
        self.directions = [-1]
        self.max_free_spaces = max_free_spaces
        self.num_free_spaces = [0 for _ in range(4)]

    
    def get_free_spaces(self):
        stack = self.get_next_nodes(self.root)

        while len(stack):
            node = stack.pop(0)
            new_nodes = self.get_next_nodes(node)
            stack.extend(new_nodes)
        
        return self.num_free_spaces

    def get_next_nodes(self, node):
        root = node is self.root
        x = node.coords[0]
        y = node.coords[1]
        node.new_nodes = []
        new_coords = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
        for i, new_coord in enumerate(new_coords):
            if (self.field[new_coord] == 0) and (new_coord not in self.visited_spaces) and (self.num_free_spaces[node.direction] < self.max_free_spaces):
                new_node = Node()
                new_node.coords = new_coord
                if root:
                    new_node.direction = i
                else:
                    new_node.direction = node.direction
                node.new_nodes.append(new_node)
                self.num_free_spaces[new_node.direction] += 1
                self.visited_spaces.append(new_coord)
                self.directions.append(new_node.direction)
        
        return node.new_nodes
    
class SurvivalTree():
    def __init__(self, game_state):
        agent_coords = game_state['self'][3]

        self.root = Node()
        self.root.direction = None
        self.root.coords = agent_coords
        self.root.time = 0
        self.root.alive = True

        self.visited = np.zeros((5, 17, 17))-1

        self.field = (game_state['field'] != 0).astype(int)
        self.walls = (game_state['field'] == -1).astype(int)
        self.bombs = game_state['bombs'].copy()

        self.bomb_maps = []
        for t in range(5):
            self.bomb_maps.append(self.calc_bomb_map(t))

        self.num_survivable_spaces = [0 for _ in range(5)]
        self.survived = [[] for _ in range(5)]

    def calc_bomb_map(self, t):

        bombs = [(b[0], b[1]-t) for b in self.bombs]
        bombs = [b for b in bombs if b[1] >= -1]

        bomb_map = np.zeros((17+6, 17+6))-2
        bomb_cooldowns = [x[1] for x in bombs]
        bomb_idx_sorted = np.flip(np.argsort(bomb_cooldowns))

        for idx in bomb_idx_sorted:
            bomb = bombs[idx]
            b_x = bomb[0][0]
            b_y = bomb[0][1]
            if np.sum(self.walls[b_x-1:b_x+2, b_y]) == 2:
                bomb_map[b_x+3, b_y:b_y+7] = bomb[1]
            elif np.sum(self.walls[b_x, b_y-1:b_y+2]) == 2:
                bomb_map[b_x:b_x+7, b_y+3] = bomb[1]
            else:
                bomb_map[b_x:b_x+7, b_y+3] = bomb[1]
                bomb_map[b_x+3, b_y:b_y+7] = bomb[1]
        return bomb_map[3:-3,3:-3]
    
    def get_free_spaces(self):
        stack = self.get_next_nodes(self.root)

        while len(stack):
            node = stack.pop(0)
            new_nodes = self.get_next_nodes(node)
            stack.extend(new_nodes)

        return self.num_survivable_spaces

    def get_next_nodes(self, node):
        root = node is self.root
        x = node.coords[0]
        y = node.coords[1]
        node.new_nodes = []
        new_coords = [(x+1, y), (x, y+1), (x-1, y), (x, y-1), (x, y)]
        for i, new_coord in enumerate(new_coords):
            if (self.field[new_coord] == 0) and node.time < 4 and node.alive:
                new_node = Node()
                new_node.coords = new_coord
                new_node.time = node.time + 1
                new_node.alive = node.alive

                if self.bomb_maps[new_node.time][new_node.coords] == -1:
                    new_node.alive = False

                if root:
                    new_node.direction = i
                else:
                    new_node.direction = node.direction

                if new_node.alive and new_node.time == 4 and new_node.coords not in self.survived[new_node.direction]:
                    self.num_survivable_spaces[new_node.direction] += 1
                    self.survived[new_node.direction].append(new_node.coords)

                if new_node.alive:
                    self.visited[new_node.time, new_node.coords[0], new_node.coords[1]] = new_node.direction
                node.new_nodes.append(new_node)
        
        return node.new_nodes
    
if __name__ == '__main__':
    states = pickle.load(open('featureset.pt', 'rb'))
    for state in states:
        test_tree = SurvivalTree(state)
        spaces = test_tree.get_free_spaces()
        
        x, y = state['self'][3]
        vis_field = state['field'].copy()
        vis_field[x, y] = 3

        for bomb in state['bombs']:
            bx, by = bomb[0]
            vis_field[bx, by] = 2

        print('D - R - U - L - W')
        print(spaces)

        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.subplot(331)
        plt.imshow(vis_field)
        plt.subplot(332)
        plt.imshow(test_tree.bomb_maps[0]-test_tree.walls)
        plt.subplot(333)
        plt.imshow(test_tree.bomb_maps[1]-test_tree.walls)
        plt.subplot(334)
        plt.imshow(test_tree.bomb_maps[2]-test_tree.walls)
        plt.subplot(335)
        plt.imshow(test_tree.bomb_maps[3]-test_tree.walls)
        plt.subplot(336)
        plt.imshow(test_tree.bomb_maps[4]-test_tree.walls)
        #plt.subplot(333)
        #plt.imshow(test_tree.visited[0]-test_tree.walls)
        #plt.subplot(334)
        #plt.imshow(test_tree.visited[1]-test_tree.walls)
        #plt.subplot(335)
        #plt.imshow(test_tree.visited[2]-test_tree.walls)
        #plt.subplot(336)
        #plt.imshow(test_tree.visited[3]-test_tree.walls)
        #plt.subplot(337)
        #plt.imshow(test_tree.visited[4]-test_tree.walls)
        plt.show()

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

class FeatureSelector():

    def __init__(self, game_state=None):
        if game_state != None:
            self.setup(game_state)


    def setup(self, game_state):
        self.game_state = game_state
        self.PADCONST = 3

        self.features = []

        self.field = self.game_state['field']
        self.walls = (self.game_state['field'] == -1).astype(int)
        self.walls_padded = np.pad(self.walls, self.PADCONST, 'edge')
        self.crates = (self.game_state['field'] == 1).astype(int)
        self.crates_padded = np.pad(self.crates, self.PADCONST, 'edge')

        self.agent_pos = self.game_state['self'][3]
        self.agent_pos_x = self.agent_pos[0]
        self.agent_pos_y = self.agent_pos[1]
        self.agent_pos_x_pad = self.agent_pos_x + self.PADCONST
        self.agent_pos_y_pad = self.agent_pos_y + self.PADCONST


    def current_step(self):
        # Current Step Number
        step_number = self.game_state['step']
        self.features.append(step_number)


    def sum_of_crates_around_agent(self):
        # Sum of crates in radius 1 around agent
        self.features.append(np.sum(self.crates_padded[self.agent_pos_x_pad-1:self.agent_pos_x_pad+2, self.agent_pos_y_pad-1:self.agent_pos_y_pad+2]))

        # Sum of self.crates in radius 2 around agent
        self.features.append(np.sum(self.crates_padded[self.agent_pos_x_pad-2:self.agent_pos_x_pad+3, self.agent_pos_y_pad-2:self.agent_pos_y_pad+3]))


    def agent_in_explosion_zone(self):
        # Agent in explosion zone (bool)
        bomb_map = np.zeros((17+2*self.PADCONST, 17+2*self.PADCONST))
        bomb_cooldowns = [x[1] for x in self.game_state['bombs'] if x[1] != 0]
        bomb_idx_sorted = np.flip(np.argsort(bomb_cooldowns))

        for idx in bomb_idx_sorted:
            bomb = self.game_state['bombs'][idx]
            b_x = bomb[0][0]
            b_y = bomb[0][1]
            if np.sum(self.walls[b_x-1:b_x+2, b_y]) == 2:
                bomb_map[b_x + self.PADCONST, b_y:b_y + self.PADCONST + 4] = bomb[1]
            elif np.sum(self.walls[b_x, b_y-1:b_y+2]) == 2:
                bomb_map[b_x:b_x + self.PADCONST + 4, b_y + self.PADCONST] = bomb[1]
            else:
                bomb_map[b_x:b_x + self.PADCONST + 4, b_y + self.PADCONST] = bomb[1]
                bomb_map[b_x + self.PADCONST, b_y:b_y + self.PADCONST + 4] = bomb[1]
        
        self.features.append(bomb_map[self.agent_pos_x_pad, self.agent_pos_y_pad])


    def crates_and_agents_explode(self):
        # Crates that explode if bomb is placed at current location
        custom_bomb_map = np.zeros((17+2*self.PADCONST, 17+2*self.PADCONST))

        b_x = self.agent_pos_x_pad
        b_y = self.agent_pos_y_pad
        if np.sum(self.walls_padded[b_x-1:b_x+2, b_y]) == 2:
            custom_bomb_map[b_x, b_y-3:b_y+4] = 1
        elif np.sum(self.walls_padded[b_x, b_y-1:b_y+2]) == 2:
            custom_bomb_map[b_x-3:b_x+4, b_y] = 1
        else:
            custom_bomb_map[b_x-3:b_x+4, b_y] = 1
            custom_bomb_map[b_x, b_y-3:b_y+4] = 1

        self.features.append(np.sum(custom_bomb_map*self.crates_padded))

        # Number of agents that explode if bomb is placed at current location (and agent doesn't move)
        agents_in_explosion_radius = 0
        for agent in self.game_state['others']:
            other_agent_pos = agent[3]
            other_agent_pos_x = other_agent_pos[0]
            other_agent_pos_y = other_agent_pos[1]
            if custom_bomb_map[other_agent_pos_x + self.PADCONST][other_agent_pos_y + self.PADCONST] == 1:
                agents_in_explosion_radius += 1
        self.features.append(agents_in_explosion_radius)


    def distance_from_bomb_and_cooldown(self):
        # Agent straight distance from next bomb (4 if bomb further away than 4) and its cooldown
        bomb_distance = 4
        bomb_cooldown = 4

        for bomb in [b for b in self.game_state['bombs'] if b[1] != 0]:
            if len(self.game_state['bombs'])==0:
                break
            else:
                if np.sum(self.walls_padded[bomb[0][0]:bomb[0][0]+2*self.PADCONST+1, bomb[0][1]]) == 2 and bomb[1]==self.agent_pos_y:
                    distance = abs(bomb[0][0] - self.agent_pos_x)
                    cooldown = bomb[1]
                    if distance < bomb_distance:
                        bomb_distance = distance
                    if cooldown < bomb_cooldown:
                        bomb_cooldown = cooldown
                elif np.sum(self.walls_padded[bomb[0][0], bomb[0][1]+2*self.PADCONST:bomb[0][1]+1]) == 2 and bomb[0]==self.agent_pos_x:
                    distance = abs(bomb[0][1] - self.agent_pos_y)
                    cooldown = bomb[1]
                    if distance < bomb_distance:
                        bomb_distance = distance
                    if cooldown < bomb_cooldown:
                        bomb_cooldown = cooldown
                elif bomb[0][0]==self.agent_pos_x:
                    distance = abs(bomb[1] - self.agent_pos_y)
                    cooldown = bomb[1]
                    if distance < bomb_distance:
                        bomb_distance = distance
                    if cooldown < bomb_cooldown:
                        bomb_cooldown = cooldown
                elif bomb[0][1]==self.agent_pos_y:
                    distance = abs(bomb[0][0] - self.agent_pos_x)   
                    cooldown = bomb[1] 
                    if distance < bomb_distance:
                        bomb_distance = distance
                    if cooldown < bomb_cooldown:
                        bomb_cooldown = cooldown

        self.features.append(bomb_distance)
        self.features.append(bomb_cooldown)


    def distance_and_rel_coords_and_crates_around_next_agent(self):
        # L1-distance to next agent
        distance_to_next_agent = 30
        closest_agent_x = self.agent_pos_x
        closest_agent_y = self.agent_pos_y
        for agent in self.game_state['others']:
            other_agent_pos = agent[3]
            other_agent_pos_x = other_agent_pos[0]
            other_agent_pos_y = other_agent_pos[1]
            distance_btw_agents = abs(other_agent_pos_x - self.agent_pos_x) + abs(other_agent_pos_y - self.agent_pos_y)
            if distance_btw_agents < distance_to_next_agent:
                distance_to_next_agent = distance_btw_agents
                closest_agent_x = other_agent_pos_x
                closest_agent_y = other_agent_pos_y
        self.features.append(distance_to_next_agent)

        # Relative coordinates of next agent
        self.features.append(closest_agent_x - self.agent_pos_x)
        self.features.append(closest_agent_y - self.agent_pos_y)

        # Crates in radius 1 around next agent
        self.features.append(np.sum(self.crates_padded[closest_agent_x+self.PADCONST-1:closest_agent_x+self.PADCONST+2, closest_agent_y+self.PADCONST-1:closest_agent_y+self.PADCONST+2]))


    def get_agent_pos(self):
        self.features.append(self.game_state['self'][3][0])
        self.features.append(self.game_state['self'][3][1])


    def coins_in_radius_3(self):
        # Coins in radius 3
        coins_in_radius = 0
        for coin in self.game_state['coins']:
            if coin[0]+self.PADCONST in range(self.agent_pos_x_pad - 3, self.agent_pos_x_pad + 4) and coin[1]+self.PADCONST in range(self.agent_pos_y_pad - 3, self.agent_pos_y_pad + 4):
                coins_in_radius += 1
        self.features.append(coins_in_radius)


    def distance_and_rel_coords_and_crates_around_coin(self):
        # L1-distance to next coin
        distance_to_next_coin = 30
        closest_coin_x = self.agent_pos_x
        closest_coin_y = self.agent_pos_y
        for coin in self.game_state['coins']:
            coin_x = coin[0]
            coin_y = coin[1]
            distance_to_coins = abs(coin_x - self.agent_pos_x) + abs(coin_y - self.agent_pos_y)
            if distance_to_coins < distance_to_next_coin:
                distance_to_next_coin = distance_to_coins
                closest_coin_x = coin_x
                closest_coin_y = coin_y
        self.features.append(distance_to_next_coin)

        # Relative coordinates of next coin
        self.features.append(closest_coin_x - self.agent_pos_x)
        self.features.append(closest_coin_y - self.agent_pos_y)

        # Crates in radius 1 around next coin
        self.features.append(np.sum(self.crates_padded[closest_coin_x+self.PADCONST-1:closest_coin_x+self.PADCONST+2, closest_coin_y+self.PADCONST-1:closest_coin_y+self.PADCONST+2]))


    def bomb_possible(self):
        # Is bomb setting possible?
        bomb_possible = self.game_state['self'][2]
        self.features.append(int(bomb_possible))

    def scores(self):
        # Agent score
        agent_score = self.game_state['self'][1]
        self.features.append(agent_score)

        # Other agents scores
        other_agent_scores = [0, 0, 0]
        for idx, agent in enumerate(self.game_state['others']):
            other_agent_scores[idx] = agent[1]
        other_agent_scores = np.sort(other_agent_scores)
        for idx, agent_score in enumerate(other_agent_scores):
            self.features.append(agent_score)


    def sum_crates(self):
        # Sum of all crates
        crate_sum = np.sum(self.crates)
        self.features.append(crate_sum)


    def agent_at_border(self):
        # Agent is at left/right border (bool)
        agent_at_leftright_border = 1 if self.agent_pos_x==1 or self.agent_pos_x==15 else 0
        # agent_at_leftright_border = 1 if (np.sum(self.walls_padded[self.agent_pos_x_pad-3:self.agent_pos_x_pad,self.agent_pos_y])==3 or np.sum(self.walls_padded[self.agent_pos_x_pad:self.agent_pos_x_pad+3, self.agent_pos_y_pad])==3) else 0
        self.features.append(agent_at_leftright_border)

        # Agent is at top/bottom border (bool)
        agent_at_topbottom_border = 1 if self.agent_pos_y==1 or self.agent_pos_y==15 else 0
        # agent_at_topbottom_border = 1 if (np.sum(self.walls_padded[self.agent_pos_x_pad,self.agent_pos_y-3:self.agent_pos_y_pad])==3 or np.sum(self.walls_padded[self.agent_pos_x_pad, self.agent_pos_y_pad:self.agent_pos_y_pad+3])==3) else 0
        self.features.append(agent_at_topbottom_border)
        # all_features.append(features)

    def agent_position_relative_to_wall(self):
        if np.sum(self.walls[self.agent_pos_x-1:self.agent_pos_x+2, self.agent_pos_y]) == 2:
            agent_pos_rel_wall = 1
        elif np.sum(self.walls[self.agent_pos_x, self.agent_pos_y-1:self.agent_pos_y+2]) == 2:
            agent_pos_rel_wall = 2
        else:
            agent_pos_rel_wall = 0
        self.features.append(agent_pos_rel_wall)

    def search_free_space(self):
        tree = MovementTree((self.agent_pos_x, self.agent_pos_y), self.game_state['field'], 7)
        self.features.extend(tree.get_free_spaces()) # gives back 4 features

    def survivable_space(self):
        tree = SurvivalTree(self.game_state)
        self.features.extend(tree.get_free_spaces()) # gives back 5 features

    def eval_coin_features(self, game_state):
        self.setup(game_state)
        #self.current_step()
        #self.sum_of_crates_around_agent()
        ## self.agent_in_explosion_zone()
        #self.crates_and_agents_explode()
        ## self.distance_from_bomb_and_cooldown()
        #self.distance_and_rel_coords_and_crates_around_next_agent()
        #self.get_agent_pos()
        #self.coins_in_radius_3()
        self.distance_and_rel_coords_and_crates_around_coin()
        #self.bomb_possible()
        #self.scores()
        #self.sum_crates()
        #self.agent_at_border()
        self.agent_position_relative_to_wall()
        self.search_free_space()
        #self.survivable_space()
        return np.array(self.features)

    def eval_singleplayer_features(self, game_state):
        self.setup(game_state)
        #self.current_step()
        self.sum_of_crates_around_agent()
        ## self.agent_in_explosion_zone()
        self.crates_and_agents_explode()
        ## self.distance_from_bomb_and_cooldown()
        #self.distance_and_rel_coords_and_crates_around_next_agent()
        #self.get_agent_pos()
        #self.coins_in_radius_3()
        self.distance_and_rel_coords_and_crates_around_coin()
        self.bomb_possible()
        #self.scores()
        self.sum_crates()
        #self.agent_at_border()
        self.agent_position_relative_to_wall()
        self.search_free_space()
        self.survivable_space()
        return np.array(self.features)

    def eval_all_features(self, game_state):
        self.setup(game_state)
        #self.current_step()
        self.sum_of_crates_around_agent()
        ## self.agent_in_explosion_zone()
        self.crates_and_agents_explode()
        ## self.distance_from_bomb_and_cooldown()
        self.distance_and_rel_coords_and_crates_around_next_agent()
        #self.get_agent_pos()
        #self.coins_in_radius_3()
        self.distance_and_rel_coords_and_crates_around_coin()
        self.bomb_possible()
        #self.scores()
        self.sum_crates()
        #self.agent_at_border()
        self.agent_position_relative_to_wall()
        self.search_free_space()
        self.survivable_space()

        return np.array(self.features)

class CircularArray:
    
    def __init__(self, shape):
        self.buffer = np.zeros(shape)
        self.populated = np.zeros(shape[0])
        self.low = 0
        self.high = 0
        self.size = shape[0]
        self.count = 0

    def isEmpty(self):
        return self.count == 0

    def isFull(self):
        return self.count == self.size
        
    def __len__(self):
        return self.count
        
    def add(self, value):
        if self.isFull():
            self.low = (self.low+1) % self.size
        else:
            self.count += 1
        self.buffer[self.high] = value
        self.populated[self.high] = 1
        self.high = (self.high + 1) % self.size
    
    def remove(self):
        if self.count == 0:
            raise Exception ("Circular Buffer is empty")
        value = self.buffer[self.low]
        self.populated[self.low] = 0
        self.low = (self.low + 1) % self.size
        self.count -= 1
        return value

    def as_array(self):
        if self.isFull():
            return self.buffer
        else:
            return self.buffer[self.populated==1]
    
    def recall(self):
        out_array = np.zeros(self.count)
        for i, item in enumerate(self):
            out_array[i] = item
        return out_array

    
    def __iter__(self):
        """Return elements in the circular buffer in order using iterator."""
        idx = self.low
        num = self.count
        while num > 0:
            yield self.buffer[idx]
            idx = (idx + 1) % self.size
            num -= 1

    def __repr__(self):
        """String representation of circular buffer."""
        if self.isEmpty():
            return 'cb:[]'

        return 'cb:[' + ','.join(map(str,self)) + ']'

def setup(self):
    if not self.train:
        self.feature_selector = FeatureSelector()
        self.actor = TreeModel(10000, (16,), (6,), 100)
        self.actor.load('actor.p')

def act(self, game_state: dict):
    if self.train:
        if game_state['round'] == 1:
            epsilon = 1
        else:
            epsilon = 0.01+0.1*np.exp(-game_state['round']/200)
    else:
        epsilon = 0.01

    if np.random.random() > epsilon:
        state, action, rot, mir = rotate_game_states([game_state], [np.array([0,0,0,0,0,0])])
        features =  self.feature_selector.eval_all_features(state[0])

        p = self.actor.predict(features.reshape((1, -1)))[0]
        #gent.logger.info(p)
        p = reverse_rotate_action(p, rot, mir)
        action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'][np.argmax(p)]
        #action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=softmax(p))

    else: 
        p = np.array([1,1,1,1,0.1,0])
        p /= np.sum(p)
        action = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT'], p=p)

    self.logger.info('Pick %s'%action)

    return action
