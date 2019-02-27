import numpy as np

def derive_state_representation(self):
    """
    From provided game_state, extract array state representation.
    :param self:
    :return: State representation in specified format
    """

    player_block = 6+17

    state = np.zeros(self.x_y_to_index(16, 16, 17, 17) + 4 * player_block + 1)

    state[-1] = self.game_state['step']




