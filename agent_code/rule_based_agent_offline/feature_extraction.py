import numpy as np
from treesearch import MovementTree

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
        if np.sum(self.walls[agent_pos_x-1:agent_pos_x+2, agent_pos_y]) == 2:
                agent_pos_rel_wall = 1
            elif np.sum(self.walls[agent_pos_x, agent_pos_y-1:agent_pos_y+2]) == 2:
                agent_pos_rel_wall = 2
            else:
                agent_pos_rel_wall = 0
        self.features.append(agent_pos_rel_wall)

    def search_free_space(self):
        tree = MovementTree((x, y), game_state['field'], 7)
        self.features.extend(tree.get_free_spaces()) # gives back 4 features


    def eval_all_features(self, game_state):
        self.setup(game_state)
        self.current_step()
        self.sum_of_crates_around_agent()
        self.agent_in_explosion_zone()
        self.crates_and_agents_explode()
        self.distance_from_bomb_and_cooldown()
        self.distance_and_rel_coords_and_crates_around_next_agent()
        self.get_agent_pos()
        self.coins_in_radius_3()
        self.distance_and_rel_coords_and_crates_around_coin()
        self.bomb_possible()
        self.scores()
        self.sum_crates()
        self.agent_at_border()
        self.agent_position_relative_to_wall()
        self.search_free_space()

        all_features = np.array(self.features)
        return all_features