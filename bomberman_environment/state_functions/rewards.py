
from settings import events
import numpy as np

def get_reward(state, player_index):

    """
    Given a game state and player index (0 to 3), return that player's reward for this step.

    :param state: Game state in array representation
    :param player_index: Player index in array (0 to 3)
    :return: Reward
    """
    debug_mode = False
    rewards = {'INVALID_ACTION': -8000,
            'KILLED_OPPONENT': 500,
            'KILLED_SELF': -2000,
            'COIN_COLLECTED': 600,
            'WAITED': -100,
            'CRATE_DESTROYED': 50,
            'GOT_KILLED': -2000}

    begin = state.shape[0]-(1 + (4 - player_index)*21)
    end = state.shape[0]-(1 + (4 - player_index - 1)*21)
    player = state[begin: end]

    reward = 0
    for event_index, multiplicity in enumerate(player[4:]):
        event = events[event_index]
        reward += rewards[event]*multiplicity if event in rewards.keys() else 0
        if multiplicity > 0 and debug_mode:

            print("---")
            print(event)
            print(reward)
        if event == 'KILLED_SELF' and multiplicity > 0:
            return rewards[event]
    if reward == 0:
        reward = -3
    return reward



