
from collections import namedtuple
import pygame
from pygame.locals import *
import logging


settings = {
    # Display
    'width': 1000,
    'height': 600,
    'gui': True,
    'fps': 15,

    # Main loop
    'update_interval': 0.1, # 0.33,
    'turn_based': False,
    'n_rounds': 1000,
    'save_replay': False,
    'make_video_from_replay': False,

    # Game properties
    'cols': 17,
    'rows': 17,
    'grid_size': 30,
    #'crate_density': 0.75,
    # For task one no creates are required
    'crate_density': 0,
    'actions': ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB'],
    'max_agents': 4,
    'max_steps': 400,
    'stop_if_not_training': False,
    'bomb_power': 3,
    'bomb_timer': 4,
    'explosion_timer': 2,

    # Rules for agents
    'timeout': 20.0,
    'reward_kill': 5,
    'reward_coin': 1,
    'reward_slow': -1,

    # User input
    'input_map': {
        K_UP: 'UP',
        K_DOWN: 'DOWN',
        K_LEFT: 'LEFT',
        K_RIGHT: 'RIGHT',
        K_RETURN: 'WAIT',
        K_SPACE: 'BOMB',
    },

    # Logging levels
    'log_game': logging.INFO,
    'log_agent_wrapper': logging.INFO,
    'log_agent_code': logging.DEBUG,
}
settings['grid_offset'] = [(settings['height'] - settings['rows']*settings['grid_size'])//2] * 2
s = namedtuple("Settings", settings.keys())(*settings.values())


events = [
    'MOVED_LEFT', #0
    'MOVED_RIGHT',
    'MOVED_UP',
    'MOVED_DOWN',
    'WAITED', #4
    'INTERRUPTED',
    'INVALID_ACTION',

    'BOMB_DROPPED', #7
    'BOMB_EXPLODED',

    'CRATE_DESTROYED', #9
    'COIN_FOUND',
    'COIN_COLLECTED',

    'KILLED_OPPONENT', #12
    'KILLED_SELF',

    'GOT_KILLED', #14
    'OPPONENT_ELIMINATED',
    'SURVIVED_ROUND',
]
e = namedtuple('Events', events)(*range(len(events)))
