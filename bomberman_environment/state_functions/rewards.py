
from settings import events
import numpy as np


event_rewards = np.zeros(len(events))
event_rewards[4] = -100     # WAITED
event_rewards[6] = -80000    # INVALID ACTION
event_rewards[10] = 0       # COIN FOUND
event_rewards[11] = 600     # COIN COLLECTED
event_rewards[12] = 500     # KILLED OPPONENT
event_rewards[13] = -2000   # KILLED SELF
event_rewards[14] = -2000   # GOT KILLED

def get_reward(state, player_index):

    """
    Given a game state and player index (0 to 3), return that player's reward for this step.

    :param state: Game state in array representation
    :param player_index: Player index in array (0 to 3)
    :return: Reward
    """

    begin = state.shape[0] - (1 + (4 - player_index) * 21)
    end = state.shape[0] - (1 + (4 - player_index - 1) * 21)
    player = state[begin: end]

    return -3 + np.sum(player[4:] * event_rewards)




