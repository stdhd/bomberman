import numpy as np
from indices import *
from rewards import *
import math


def create_window(array_source, radius, agents_count):
    ZEILEN = 17
    WALL = 2
    CRATE = 1
    FREE = 0
    COIN = -1
    PLAYER = 3
    BOMB = 4
    EXPLOSION = 5
    AGENT = 6
    INDEX_LENGTH = 176

    row_length = radius * 2 + 1
    array_target = np.zeros([agents_count, row_length, row_length])
    rewards = np.zeros([agents_count])
    events = np.zeros([agents_count])

    # Add Player and Bomb coordinates
    startX = INDEX_LENGTH
    for l in range(agents_count):
        mx, my = index_to_x_y(int(daten[int(startX)]), ZEILEN,
                              ZEILEN)  # Koordinaten des Spielers, Mittelpunkt des Ausschnitts
        borderX = mx + radius
        borderY = my + radius

        start = INDEX_LENGTH
        for i in range(agents_count):
            agentX, agentY = index_to_x_y(int(daten[int(start)]), ZEILEN, ZEILEN)
            # Set agents' positions
            if not (agentX > borderX or agentY > borderY or agentX < (mx - radius) or agentY < (my - radius)):
                array_target[l, int(agentY - (my - radius)), int(agentX - (mx - radius))] = AGENT
                start = start

            # Write bombs
            start += 2
            if not daten[int(start)] == 0:
                bombeX, bombeY = index_to_x_y(int(daten[int(start)]), ZEILEN, ZEILEN)
                if not (bombeX > borderX or bombeY > borderY or bombeX < (mx - radius) or bombeY < (my - radius)):
                    start = start
                    array_target[l, int(bombeY - (my - radius)), int(bombeX - (mx - radius))] = BOMB
            start = start + 19

            # Write rewards
        # rewards[l] =
        daten[int(startX + 1)]

        # Write event
        events[l] = np.argmax(daten[int(startX + 4): int(startX + 8)])

        # print(np.argmax(temp_events))

        for i in range(int(math.pow(row_length, 2))):

            xt = i % row_length
            yt = math.floor(i / (row_length))
            x = xt + (mx - radius)
            y = yt + (my - radius)

            if array_target[l, int(yt), int(xt)] == 0:

                # print(y,x)
                try:
                    index = x_y_to_index(int(x), int(y), ZEILEN, ZEILEN)
                    start = start
                    # print(int(y),int(x))
                    array_target[l, int(yt), int(xt)] = daten[int(index)]
                except Exception as e:
                    array_target[l, yt, xt] = WALL
                    # print("WALL")
                    start = start
            # else:
            # print("Kollision " + str(int(y)) + " " + str(int(x)) )
            # print(int(yt),int(xt))

        startX = startX + 21
        # print("---")
    return array_target, events, rewards

daten = np.load("test_daten.npy")[100]

get_reward(daten, 1)