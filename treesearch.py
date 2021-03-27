import numpy as np
import matplotlib.pyplot as plt
import pickle

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
    
def vis_spaces(visited, dirs):
    field = np.zeros((17, 17))-2
    for coords, dir in zip(visited, dirs):
        field[coords] = dir
    return field

if __name__ == '__main__':
    states = pickle.load(open('featureset.pt', 'rb'))
    for state in states:
        field = state['field']
        x, y = state['self'][3]

        test_tree = MovementTree((x, y), field, 7)
        spaces = test_tree.get_free_spaces()
        
        vis_field = field.copy()
        vis_field[x, y] = 2
        print('R - L - D - U')
        print(spaces)

        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.subplot(221)
        plt.imshow(vis_field)
        plt.subplot(222)
        plt.imshow(vis_spaces(test_tree.visited_spaces, test_tree.directions))
        plt.show()