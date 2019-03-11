
from settings import events
import numpy as np

def get_reward(state, player_index):

    """
    Given a game state and player index (0 to 3), return that player's reward for this step.

    :param state: Game state in array representation
    :param player_index: Player index in array (0 to 3)
    :return: Reward
    """

    rewards = {'INVALID_ACTION': -10,
            'KILLED_OPPONENT': 500,
            'KILLED_SELF': -1000,
            'COIN_COLLECTED': 100}

    begin = state.shape[0]-(1 + (4 - player_index)*21)
    end = state.shape[0]-(1 + (4 - player_index - 1)*21)
    player = state[begin: end]

    reward = 0

    for event_index, count in player[4:]:  # index corresponds to event, value at index is multiplicity of event

        event = events[event_index]

        reward += rewards[event]*count if event in rewards.keys() else 0

    return reward



