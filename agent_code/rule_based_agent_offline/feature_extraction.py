import numpy as np
import pickle

featurepickle = open("./featureset.pt", "rb")
labelpickle = open("./labelset.pt", "rb")

gamestates = pickle.load(featurepickle)
labels = pickle.load(labelpickle)

all_features = []

for idx, game_state in enumerate(gamestates):

	features = []
	
	field = game_state['field']
	walls = (game_state['field'] == -1).astype(int)
	crates = (game_state['field'] == 1).astype(int)

	agent_pos = game_state['self'][3]
	agent_pos_x = agent_pos[0]
	agent_pos_y = agent_pos[1]
	# print(field)

	# print(walls)

	# Current Step Number
	step_number = game_state['step']
	features.append(step_number)

	# Sum of crates in radius 1 around agent


	# Sum of crates in radius 2 around agent


	# Agent in explosion zone (bool)


	# Agent straight distance from next bomb (4 if bomb further away than 4)


	# L1-distance to next agent


	# Coins in radius 3


	# L1-distance to next coin


	# Is bomb setting possible?
	bomb_possible = game_state['self'][2]
	features.append(bomb_possible)

	# Agent score
	agent_score = game_state['self'][1]
	features.append(agent_score)

	# Other agents scores
	for agent in game_state['others']:
		features.append(agent[1]) 

	# Lowest countdown for bomb that's less than 4 spaces away in straight line


	# Sum of all crates


	# Crates in radius 1 around next coin


	# Crates in radius 1 around next agent


	# Space free of crates in each direction


	# Crates that explode if bomb is placed at current location


	# Agents that explode if bomb is placed at current location (and agent doesn't move)


	# Chance of survival if bomb places here (bool)


	# Relative coordinates of next coin


	# Relative coordinates of next agent


	# Agent is at left/right border (bool)


	# Agent is at top/bottom border (bool)



featurepickle.close()
labelpickle.close()