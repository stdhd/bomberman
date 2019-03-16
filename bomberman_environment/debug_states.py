

import numpy as np
from os import listdir, getcwd
from os.path import isfile, join
from agent_code.observation_object import ObservationObject

import settings

def main():
    """
    Load game files and step through them, drawing the game as a window and showing agent events.
    :return:
    """

    directory = 'data/games/one_player_only_coins'

    for file in [f for f in listdir(directory) if isfile(join(directory, f))]:
        # go through files
        if file == ".DS_Store":
            continue
        game = np.load(directory+"/"+file)

        obs = ObservationObject(0, None, [])

        for step in game:
            obs.set_state(step)
            window = obs._make_window(8, 8, 8)

            events = []

            for i in range(4):
                events.append([False for i in range(17)])
                for j, count in enumerate(step[obs.board.shape[0] + 4 + i*21 : obs.board.shape[0] + 4 + i*21 + 17]):
                    if count != 0:
                        events[i][j] = True

                events[i] = np.array(settings.events)[np.array(events[i])]

            pass

if __name__ == '__main__':
    main()
