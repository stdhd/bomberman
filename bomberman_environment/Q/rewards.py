
from settings import events


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

    player = state[-(1 + (4 - player_index)*21): -(1 + (4 - player_index + 1)*21)]

    step_events = events[player[4:]]

    reward = 0

    for ev in step_events:
        reward += rewards[ev] if ev in rewards else 0

    return reward



