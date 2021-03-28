import numpy as np
import matplotlib.pyplot as plt
import pickle

class Node():
    pass

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