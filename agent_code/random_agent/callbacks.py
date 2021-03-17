import numpy as np
from functions import handmade_features
import pickle

def setup(self):
    np.random.seed()


def act(agent, game_state: dict):
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
