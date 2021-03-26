import numpy as np
import pickle

featurepickle = open("./featureset.pt", "rb")
labelpickle = open("./labelset.pt", "rb")

features = pickle.load(featurepickle)
labels = pickle.load(labelpickle)

for idx, game_state in enumerate(features):

	# print(idx, game_state)
	field = game_state['field']
	walls = (game_state['field'] == -1).astype(int)
	crates = (game_state['field'] == 1).astype(int)

	agent_pos = game_state['self'][3]
	agent_pos_x = agent_pos[0]
	agent_pos_y = agent_pos[1]
	# print(field)

	# print(walls)

















featurepickle.close()
labelpickle.close()