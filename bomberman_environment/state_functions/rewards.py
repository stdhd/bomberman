
from settings import events
import numpy as np


event_rewards = np.zeros(len(events))
event_rewards[4] = -2       # WAITED
event_rewards[6] = -10000    # INVALID ACTION
event_rewards[9] = 10          # DESTROYED CRATES
event_rewards[10] = 0       # COIN FOUND
event_rewards[11] = 100     # COIN COLLECTED
event_rewards[12] = 500     # KILLED OPPONENT
event_rewards[14] = -600   # GOT KILLED

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
    #
    # if player[4:][13] > 0:
    #     print("!!!!")

    return np.sum(player[4:] * event_rewards)




