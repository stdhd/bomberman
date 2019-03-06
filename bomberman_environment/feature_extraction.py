import numpy as np

arena_size = 17

observation = np.load('C:/Repositories/Master/bomberman/JAKOB_bomberman/bomberman_environment/data/games/2019-03-02_13-05-43_1.npy')
print(observation.shape)
arena = observation[0, 0:176]
player1 = observation[0, 176:197]
player2 = observation[0, 197:218]
player3 = observation[0, 218:239]
player4 = observation[0, 239:260]

print(arena)
print(player1)
print(player2)
print(player3)
print(player4)

# Player 

