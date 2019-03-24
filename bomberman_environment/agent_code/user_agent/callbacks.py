
import numpy as np
from time import sleep

from agent_code.observation_object import ObservationObject
from state_functions.state_representation import derive_state_representation

def setup(agent):
    agent.obs_object = ObservationObject(1, ['d_closest_coin_dir',
                                            'd_closest_safe_field_dir',
                                            'me_has_bomb',
                                            'dead_end_detect',
                                            'd4_is_safe_to_move_a_l',
                                            'd4_is_safe_to_move_b_r',
                                            'd4_is_safe_to_move_c_u',
                                            'd4_is_safe_to_move_d_d',
                                            'd_best_bomb_dropping_dir',
                                            # 'd_closest_enemy_dir'
                                            # 'd_closest_crate_dir',
                                            ], None)


def act(agent):

    rep = derive_state_representation(agent)
    agent.obs_object.set_state(rep)
    observation = agent.obs_object.create_observation(np.array([0]))[0]
    agent.logger.debug("\n"+str(observation[:agent.obs_object.window_size**2].reshape(
            (agent.obs_object.window_size, agent.obs_object.window_size)).T))
    agent.logger.debug("Features: " + str(observation[agent.obs_object.window_size**2:]))
       
    
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
    #agent.logger.info(arena)

    # agent.logger.info(step)
    # agent.logger.info(f'Self: {x, y, name, bombs_left, score}')
    # agent.logger.info(f'Bombs: {bombs}')
    # agent.logger.info(f'Others: {others}')

def reward_update(agent):
    pass

def learn(agent):
    pass
