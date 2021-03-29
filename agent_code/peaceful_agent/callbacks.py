import numpy as np


def setup(self):
    np.random.seed()
    self.score = []
    self.prev_step = 0
    self.prev_score = 0


def act(agent, game_state: dict):
    agent.logger.info('Pick action at random, but no bombs.')
    if agent.prev_step > game_state['step']:
        agent.score.append(agent.prev_score)
    
    agent.logger.info(np.mean(agent.score))
    
    agent.prev_step = game_state['step']
    agent.prev_score = game_state['self'][1]
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])
