import numpy as np


def setup(self):
    np.random.seed()


def act(agent, game_state: dict):
    agent.logger.info('Pick action at random, but no bombs.')
    agent.logger.info(game_state['bombs'])
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'])
