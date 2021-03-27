import numpy as np
import pickle

PADCONST = 3

featurepickle = open("./featureset.pt", "rb")
labelpickle = open("./labelset.pt", "rb")

gamestates = pickle.load(featurepickle)
labels = pickle.load(labelpickle)

all_features = []


for idx, game_state in enumerate(gamestates):

	if idx == 2:
		break
	features = []
	
	field = game_state['field']
	walls = (game_state['field'] == -1).astype(int)
	walls_padded = np.pad(walls, PADCONST, 'edge')
	crates = (game_state['field'] == 1).astype(int)
	crates_padded = np.pad(crates, PADCONST, 'edge')

	print(field, "\n", walls, "\n", walls_padded, "\n", crates, "\n", crates_padded)

	agent_pos = game_state['self'][3]
	agent_pos_x = agent_pos[0]
	agent_pos_y = agent_pos[1]
	agent_pos_x_pad = agent_pos_x + PADCONST
	agent_pos_y_pad = agent_pos_y + PADCONST


	# Current Step Number
	step_number = game_state['step']
	features.append(step_number)

	# Sum of crates in radius 1 around agent
	features.append(np.sum(crates_padded[agent_pos_x_pad-1:agent_pos_x_pad+1, agent_pos_y_pad-1:agent_pos_y_pad+1]))

	# Sum of crates in radius 2 around agent
	features.append(np.sum(crates_padded[agent_pos_x_pad-2:agent_pos_x_pad+2, agent_pos_y_pad-2:agent_pos_y_pad+2]))

	# Agent in explosion zone (bool)
	bomb_map = np.zeros((17+2*PADCONST, 17+2*PADCONST))
    bomb_cooldowns = [x[1] for x in game_state['bombs']]
    bomb_idx_sorted = np.flip(np.argsort(bomb_cooldowns))

    for idx in bomb_idx_sorted:
        bomb = game_state['bombs'][idx]
        b_x = bomb[0][0]
        b_y = bomb[0][1]
        if np.sum(walls[b_x-1:b_x+1, b_y]) == 2:
        	bomb_map[b_x + PADCONST, b_y:b_y + PADCONST + 4] = bomb[1]
        elif np.sum(walls[b_x, b_y-1, b_y+1]) == 2:
        	bomb_map[b_x:b_x + PADCONST + 4, b_y + PADCONST] = bomb[1]
        else:
	        bomb_map[b_x:b_x + PADCONST + 4, b_y + PADCONST] = bomb[1]
	        bomb_map[b_x + PADCONST, b_y:b_y + PADCONST + 4] = bomb[1]
    
    features.append(bomb_map[agent_pos_x_pad, agent_pos_y_pad])

	# Crates that explode if bomb is placed at current location
	custom_bomb_map = np.zeros(17+2*PADCONST, 17+2*PADCONST)

	b_x = agent_pos_x_pad
	b_y = agent_pos_y_pad
	if np.sum(walls[b_x-1:b_x+1, b_y]) == 2:
        	custom_bomb_map[b_x + PADCONST, b_y:b_y + PADCONST + 4] = 1
        elif np.sum(walls[b_x, b_y-1, b_y+1]) == 2:
        	custom_bomb_map[b_x:b_x + PADCONST + 4, b_y + PADCONST] = 1
        else:
	        custom_bomb_map[b_x:b_x + PADCONST + 4, b_y + PADCONST] = 1
	        custom_bomb_map[b_x + PADCONST, b_y:b_y + PADCONST + 4] = 1

	features.append(np.sum(custom_bomb_map*crates_padded))

	# Number of agents that explode if bomb is placed at current location (and agent doesn't move)
	for agent in game_state['others']:
		other_agent_pos = agent[3]
		other_agent_pos_x = other_agent_pos[0]
		other_agent_pos_y = other_agent_pos[1]
		agents_in_explosion_radius = 0
		if custom_bomb_map[other_agent_pos_x][other_agent_pos_y] == 1:
			agents_in_explosion_radius += 1
	features.append(agents_in_explosion_radius)

	# Agent straight distance from next bomb (4 if bomb further away than 4)


	# L1-distance to next agent
	distance_to_next_agent = 30
	closest_agent_x = 0
	closest_agent_y = 0
	for agent in game_state['others']:
		other_agent_pos = agent[3]
		other_agent_pos_x = other_agent_pos[0]
		other_agent_pos_y = other_agent_pos[1]
		distance_btw_agents = abs(other_agent_pos_x - agent_pos_x) + abs(other_agent_pos_y - agent_pos_y)
		if distance_btw_agents < distance_to_next_agent:
			distance_to_next_agent = distance_btw_agents
			closest_agent_x = other_agent_pos_x
			closest_agent_y = other_agent_pos_y
	features.append(distance_to_next_agent)

	# Relative coordinates of next agent
	features.append(closest_agent_x - agent_pos_x)
	features.append(closest_agent_y - agent_pos_y)

	# Coins in radius 3


	# L1-distance to next coin
	distance_to_next_coin = 30
	closest_coin_x = 0
	closest_coin_y = 0
	for coin in game_state['coins']:
		coin_x = coin[0]
		coin_y = coin[1]
		distance_to_coins = abs(coin_x - agent_pos_x) + abs(coin_y - agent_pos_y)
		if distance_to_coins < distance_to_next_coin:
			distance_to_next_coin = distance_to_coins
			closest_coin_x = coin_x
			closest_coin_y = coin_y
	features.append(distance_to_next_coin)

	# Relative coordinates of next coin
	features.append(closest_coin_x - agent_pos_x)
	features.append(closest_coin_y - agent_pos_y)

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
	crate_sum = np.sum(crates)
	features.append(crate_sum)

	# Crates in radius 1 around next coin
	features.append(np.sum(crates_padded[closest_coin_x+PADCONST-1:closest_coin_x+PADCONST+1, closest_coin_y+PADCONST-1:closest_coin_y+PADCONST+1]))

	# Crates in radius 1 around next agent
	features.append(np.sum(crates_padded[closest_agent_x+PADCONST-1:closest_agent_x+PADCONST+1, closest_agent_y+PADCONST-1:closest_agent_y+PADCONST+1]))

	# Space free of crates in each direction


	# Agent is at left/right border (bool)
	agent_at_leftright_border = 1 if (np.sum(walls_padded[agent_pos_x_pad-3:agent_pos_x_pad,agent_pos_y])==3 or np.sum(walls_padded[agent_pos_x_pad:agent_pos_x_pad+3, agent_pos_y_pad])==3) else 0
	features.append(agent_at_leftright_border)

	# Agent is at top/bottom border (bool)
	agent_at_leftright_border = 1 if (np.sum(walls_padded[agent_pos_x_pad,agent_pos_y-3:agent_pos_y_pad])==3 or np.sum(walls_padded[agent_pos_x_pad, agent_pos_y_pad:agent_pos_y_pad+3])==3) else 0
	features.append(agent_at_leftright_border)


featurepickle.close()
labelpickle.close()