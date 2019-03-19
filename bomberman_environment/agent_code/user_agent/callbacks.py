
import numpy as np
from time import sleep


def setup(agent):
    pass

def act(agent):
    agent.logger.info('Pick action according to pressed key')
    agent.next_action = agent.game_state['user_input']
    arena = agent.game_state['arena']
    x, y, name, bombs_left, score = agent.game_state['self']
    bombs = agent.game_state['bombs']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b,s) in agent.game_state['others']]
    coins = agent.game_state['coin_locs']
    explosions = agent.game_state['explosions']
    step = agent.game_state['step']

    agent.logger.info(others)
    agent.logger.info(x, y)
    agent.logger.info(arena)

    # agent.logger.info(step)
    # agent.logger.info(f'Self: {x, y, name, bombs_left, score}')
    # agent.logger.info(f'Bombs: {bombs}')
    # agent.logger.info(f'Others: {others}')

def reward_update(agent):
    pass

def learn(agent):
    pass
