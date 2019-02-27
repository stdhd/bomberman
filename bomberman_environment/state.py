import numpy as np

def derive_state_representation(self):
    """
    From provided game_state, extract array state representation.

    Final state format specification in environment_save_states.py
    :param self:
    :return: State representation in specified format
    """

    player_block = 4+17

    state = np.zeros(self.x_y_to_index(16, 16, 17, 17) + 4 * player_block + 1)

    state[-1] = self.game_state['step']

    arena = self.game_state['arena']

    explosions = self.game_state['explosions']

    for x in range(arena.shape[0] - 1):
        if x == 0:
            continue
        for y in range(arena.shape[1] - 1):
            if y == 0:
                continue
            if (x + 1) * (y + 1) % 2 == 1:
                state[self.x_y_to_index(x, y, 17, 17)] = -1

            elif explosions[x, y] != 0:







